#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:22:55 2026

@author: Marcel Hesselberth
"""

import numpy as np
import networkx as nx
import re
import matplotlib.pyplot as plt


class Netlist:
    """
    Class for reading a netlist and analyzing its multigraph G.
    """

    priorities = {'V': 1, 'C': 10, 'R': 100, 'L': 1000, 'I': 10000}

    def __init__(self, netlist_str):
        """
        Constructor. Set up the graph G and parse the netlist.
        
        A netlist entry is either a comment, an LTI component of type
        R, C or L or a source of type V or I. Entries are of the form
        R1 pnode nnode value ; drawing hints # comments
        The component name is R1, its type is R, pnode and nnode are the
        positive and negative nodes and value must be a string that can be
        converted to a float. Node names are case sensitive.
        """
        self.G = nx.MultiGraph()
        self.components = []
        self.node_map = {'0': 0}
        self.node_names = ['0']
        self.parse(netlist_str)

    def _get_node_id(self, name):
        """
        Get the internal id of a netlist node.
        This guarantees that for matrix processing the nodes are consecutively
        numbered.

        Returns: int, the unique id of the node.
        """
        if name not in self.node_map:
            new_id = len(self.node_names)
            self.node_map[name] = new_id
            self.node_names.append(name)
        return self.node_map[name]

    def parse(self, netlist_str):
        """
        Parse the netlist according to the usual format (described above).
        """
        lines = netlist_str.strip().split('\n')
        for i, line in enumerate(lines):
            s = line.split('#', maxsplit=1)
            clean = s[0]
            comment = s[1] if len(s) > 1 else ""
            s = clean.split(';', maxsplit=1)
            clean = s[0]
            hint = s[1] if len(s) > 1 else ""
            clean = clean.strip()
            if not clean: continue
            parts = re.split(r'\s+', clean)
            if len(parts) < 4:
                print(f"Skipping line {line}")
                continue
            name, n1_str, n2_str, val_str = parts[:4]
            n1, n2 = self._get_node_id(n1_str), self._get_node_id(n2_str)
            self.components.append({'name': name, 
                                    'type': name[0].upper(), 
                                    'n1': n1, 
                                    'n2': n2, 
                                    'val': float(val_str), 
                                    'hint': hint,
                                    'comment': comment})
            self.G.add_edge(n1, n2, name=name, type=name[0].upper(), 
                            weight=self.priorities.get(name[0].upper(), 500))

    def get_nVL(self):
        return len(self.node_names) - 1, \
               [c['name'] for c in self.components if c['type'] in ('V', 'L')]
    
    def check_topology(self, svd_rank=None):
        results = {"errors": [], "warnings": [], "info": {}}
        num_nodes = len(self.node_names)
        
        # V, R, C determine structure / connectivity to GND
        rigid_graph = nx.Graph()
        rigid_graph.add_nodes_from(range(num_nodes))
        for c in self.components:
            if c['type'] in ('V', 'R', 'C'):
                rigid_graph.add_edge(c['n1'], c['n2'], 
                                     name=c['name'], type=c['type'])
    
        # ERRORS (physically impossible)
        # Parallel voltage sources
        v_graph = nx.MultiGraph()
        for c in [comp for comp in self.components if comp['type'] == 'V']:
            if v_graph.has_edge(c['n1'], c['n2']):
                results["errors"].append(f"Conflict: Parallel Voltage Sources between nodes {self.node_names[c['n1']]} and {self.node_names[c['n2']]}")
            v_graph.add_edge(c['n1'], c['n2'])
    
        # Series current sources / floating parts (IL-Cutsets)
        if not nx.is_connected(rigid_graph):
            islands = list(nx.connected_components(rigid_graph))
            for island in islands:
                if 0 not in island:
                    nodes = [self.node_names[n] for n in island]
                    # Check of dit eiland alleen door I-bronnen wordt gevoed (Error)
                    results["errors"].append(f"Floating Section / I-Source Conflict at nodes: {nodes}")
    
        # WARNINGS
        # VC-Loops
        vc_graph = nx.Graph()
        for c in self.components:
            if c['type'] in ('V', 'C'):
                vc_graph.add_edge(c['n1'], c['n2'])
        vc_loops = len(nx.cycle_basis(vc_graph))
        if vc_loops > 0:
            results["warnings"].append(f"Order reduction: {vc_loops} VC-loop(s) detected. SVD rank will be lower than #L+#C.")
    
        # IL-Cutsets
        il_cutsets = nx.number_connected_components(rigid_graph) - 1
        if il_cutsets > 0:
            results["warnings"].append(f"Order reduction: {il_cutsets} IL-cutset(s) detected.")
    
        # SVD rank verification
        num_lc = len([c for c in self.components if c['type'] in ('L', 'C')])
        theoretical_order = max(0, num_lc - vc_loops - il_cutsets)
        
        results["info"] = {
            "num_lc": num_lc,
            "theoretical_order": theoretical_order,
            "vc_loops": vc_loops,
            "il_cutsets": il_cutsets
        }
    
        if svd_rank is not None:
            if svd_rank != theoretical_order:
                results["warnings"].append(f"Rank Mismatch: SVD rank ({svd_rank}) != Theoretical ({theoretical_order}). Check solver tolerance.")
        
        return results

test_netlist = """
# VC-Loop: V1, C1 en C2 vormen een loop. 
# Theoretisch: 2 condensatoren - 1 loop = 1 onafhankelijke state.
V1 1 0 10
C1 1 2 1e-6
C2 2 0 1e-6

# IL-Cutset: Knooppunt 3 is alleen verbonden met I1 en L1.
# Dit creëert een 'eiland' {3} dat geen R, V of C pad naar GND heeft.
R1 1 0 1000
I1 1 3 0.1
L1 3 0 1e-3
"""

# Parser en Checks
nl = Netlist(test_netlist)

# 1. Run de topologische check
errors = nl.check_topology()
print("Topological Errors:", errors)

# 2. Simuleer SVD rank bepaling (hypothetisch svd_rank resultaat uit jouw solver)
# In dit netwerk: 
# LC elementen = 3 (C1, C2, L1)
# VC loops = 1 (V1-C1-C2)
# IL cutsets = 1 (Node 3 is een eiland zonder V,R,C naar GND)
# Verwachte rank = 3 - 1 - 1 = 1
svd_rank_result = 1 

# 3. Verifieer met de nieuwe methode


serie_conflict_netlist = """
# Serie-conflict: I1 en I2 staan in serie. 
# Node 2 heeft geen ander pad, dus KCL zegt: I1 - I2 = 0.
# Maar 1A - 2A != 0.
V1 1 0 10
I1 1 2 1.0
I2 2 0 2.0

# Voeg een state toe om te zien of de rank beïnvloed wordt
#C1 1 0 1e-6
"""

nl = Netlist(serie_conflict_netlist)

# 1. Run de verbeterde topologische check
errors = nl.check_topology()
print("Topological Errors:")
for e in errors:
    print(f" - {e}")

# 2. Verwachte SVD Rank analyse:
# States: 1 (C1)
# VC-loops: 0
# IL-cutsets: 1 (Node 2 is een eiland verbonden door enkel I-bronnen)
# Theoretische Orde: 1 (C1) - 0 (VC) - 1 (IL) = 0

nl = Netlist(test_netlist)

# 2. (Voer hier je SVD solver uit om de 'rank' te bepalen)
# Stel: rank = np.sum(s > 1e-13)
svd_rank_found = 3

# 3. Voer de analyse uit
analysis = nl.check_topology(svd_rank=svd_rank_found)

# 4. Print de resultaten overzichtelijk
print("="*30)
print(" TOPOLOGY ANALYSIS ")
print("="*30)

# Toon Info
info = analysis["info"]
print(f"Systeem Orde:")
print(f" - L/C Componenten: {info['num_lc']}")
print(f" - VC-Loops:       {info['vc_loops']}")
print(f" - IL-Cutsets:     {info['il_cutsets']}")
print(f" - Theoretisch:    {info['theoretical_order']}")
print(f" - SVD Rank:       {svd_rank_found}")

# Toon Waarschuwingen (Geel/Oranje in gedachten)
if analysis["warnings"]:
    print("\nWarnings:")
    for w in analysis["warnings"]:
        print(f" [!] {w}")

# Toon Fouten (Rood)
if analysis["errors"]:
    print("\nErrors:")
    for e in analysis["errors"]:
        print(f" [X] {e}")
    print("\nBeware: SVD-shunt (1e-16) can mask these errors.")

print("="*30)
