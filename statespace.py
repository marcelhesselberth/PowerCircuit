#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:22:55 2026

@author: Marcel Hesselberth
"""

from netlist import Netlist
import numpy as np
import matplotlib.pyplot as plt

    
class StateSpace:
    def __init__(self, parser):
        num_nodes, branches = parser.get_nVL()
        dim = num_nodes + len(branches)
        G_mat, C_mat = np.zeros((dim, dim)), np.zeros((dim, dim))
        
        v_srcs = [c for c in parser.components if c['type'] == 'V']
        i_srcs = [c for c in parser.components if c['type'] == 'I']
        u_labels = [c['name'] for c in (v_srcs + i_srcs)]
        B_mna = np.zeros((dim, len(u_labels)))

        # 1. Build MNA Matrices
        v_idx = 0
        for c in parser.components:
            n1, n2, val, name = c['n1'], c['n2'], c['val'], c['name']
            if c['type'] == 'R':
                self._stamp_g(G_mat, n1, n2, 1.0/val)
            elif c['type'] == 'C':
                self._stamp_g(C_mat, n1, n2, val)
            elif c['type'] == 'L':
                row = num_nodes + branches.index(name)
                if n1 > 0: G_mat[row, n1-1] = G_mat[n1-1, row] = 1
                if n2 > 0: G_mat[row, n2-1] = G_mat[n2-1, row] = -1
                C_mat[row, row] = -val
            elif c['type'] == 'V':
                row = num_nodes + branches.index(name)
                if n1 > 0: G_mat[row, n1-1] = G_mat[n1-1, row] = 1
                if n2 > 0: G_mat[row, n2-1] = G_mat[n2-1, row] = -1
                B_mna[row, v_idx] = 1; v_idx += 1
            elif c['type'] == 'I':
                if n1 > 0: B_mna[n1-1, len(v_srcs) + i_srcs.index(c)] = -1
                if n2 > 0: B_mna[n2-1, len(v_srcs) + i_srcs.index(c)] = 1

        # 2. SVD Reduction (Descriptor to State Space)
        # Gebruik een zeer kleine shunt voor zwevende nodes (numerieke GND referentie)
        G_mod = G_mat + np.eye(dim) * 1e-16
        U, s, Vh = np.linalg.svd(C_mat)
        rank = np.sum(s > 1e-13)
        
        U1, U2 = U[:, :rank], U[:, rank:]
        S1_inv = np.diag(1.0/s[:rank]) if rank > 0 else np.zeros((0,0))
        V1, V2 = Vh[:rank, :].T, Vh[rank:, :].T
        
        M_inv = np.linalg.inv(U2.T @ (-G_mod) @ V2)
        T_x = V1 - V2 @ M_inv @ U2.T @ (-G_mod) @ V1 if rank > 0 else np.zeros((dim, 0))
        T_u = -V2 @ M_inv @ U2.T @ B_mna

        self.A = S1_inv @ U1.T @ ((-G_mod) - (-G_mod) @ V2 @ M_inv @ U2.T @ (-G_mod)) @ V1 if rank > 0 else np.zeros((0,0))
        self.B = S1_inv @ U1.T @ (B_mna - (-G_mod) @ V2 @ M_inv @ U2.T @ B_mna) if rank > 0 else np.zeros((0, len(u_labels)))
        
        # 3. Output Construction
        #self.y_labels = [f"v_n{i+1}" for i in range(num_nodes)] + [f"i_{b}" for b in branches]
        # Use parser.node_names to get the original strings (skipping '0' at index 0)
        self.y_labels = [f"v_{parser.node_names[i+1]}" for i in range(num_nodes)] + [f"i_{b}" for b in branches]

        self.C_ss, self.D_ss = T_x, T_u
        self.u_labels, self.x_labels = u_labels, [f"x{i}" for i in range(rank)]

        # Extra outputs: Stromen door R en C
        for c in parser.components:
            if c['type'] in ('R', 'C'):
                row = np.zeros(dim)
                if c['n1'] > 0: row[c['n1']-1] = 1
                if c['n2'] > 0: row[c['n2']-1] = -1
                if c['type'] == 'R':
                    self.C_ss = np.vstack([self.C_ss, (row @ T_x) / c['val']])
                    self.D_ss = np.vstack([self.D_ss, (row @ T_u) / c['val']])
                    self.y_labels.append(f"i_{c['name']}")
                else: # i_C = C * dv/dt = C * (C_row * (Ax + Bu))
                    self.C_ss = np.vstack([self.C_ss, c['val'] * (row @ T_x @ self.A)])
                    self.D_ss = np.vstack([self.D_ss, c['val'] * (row @ T_x @ self.B)])
                    self.y_labels.append(f"i_{c['name']}")

    def _stamp_g(self, G, n1, n2, g):
        if n1 > 0: G[n1-1, n1-1] += g
        if n2 > 0: G[n2-1, n2-1] += g
        if n1 > 0 and n2 > 0: G[n1-1, n2-1] -= g; G[n2-1, n1-1] -= g

    def get_transfer_function(self, out_label, in_label, frequencies):
        """
        Calculates H(jw) = C(jwI - A)^-1 * B + D
        """
        out_idx = self.y_labels.index(out_label)
        in_idx = self.u_labels.index(in_label)
        
        # Select the specific row of C and column of B
        C_i = self.C_ss[out_idx, :].reshape(1, -1)
        B_j = self.B[:, in_idx].reshape(-1, 1)
        D_ij = self.D_ss[out_idx, in_idx]
        
        responses = []
        I = np.eye(len(self.x_labels))
        
        for f in frequencies:
            s = 2j * np.pi * f
            if len(self.x_labels) > 0:
                # Solve (sI - A) * X = B_j
                term = np.linalg.solve(s * I - self.A, B_j)
                h_s = (C_i @ term)[0,0] + D_ij
            else:
                # Pure algebraic case
                h_s = D_ij
            responses.append(h_s)
        return np.array(responses)
    

def simulate(ss, t, u_func):
    dt = t[1] - t[0]
    num_steps = len(t)
    num_inputs = len(ss.u_labels)
    u_vals = np.zeros((num_inputs, num_steps))
    for i, label in enumerate(ss.u_labels):
        u_vals[i, :] = u_func(t, label)

    x = np.zeros(len(ss.x_labels))
    history = []

    if len(x) > 0:
        # Trapezoidal Rule matrices
        I = np.eye(len(x))
        LHS = I - (dt / 2.0) * ss.A
        RHS_mat = I + (dt / 2.0) * ss.A
        B_term = (dt / 2.0) * ss.B

        for i in range(num_steps):
            u_now = u_vals[:, i]
            if i > 0:
                u_prev = u_vals[:, i-1]
                # Trapezoidal step: x(k+1) = LHS^-1 * (RHS*x(k) + B/2 * (u(k) + u(k+1)))
                RHS = RHS_mat @ x + B_term @ (u_prev + u_now)
                x = np.linalg.solve(LHS, RHS)
            
            y_now = ss.C_ss @ x + ss.D_ss @ u_now
            history.append(y_now)
    else:
        # Purely algebraic
        for i in range(num_steps):
            history.append(ss.D_ss @ u_vals[:, i])

    return np.array(history)

    
netlist_str = """
V1 3 0 1
V2 1 3 1
#L1 1 2 1e-6
C1 2 0 1e-4
R1 1 2 1
"""

netlist = Netlist(netlist_str)
ss = StateSpace(netlist)

t = np.linspace(0, 10e-3, 5000)

def my_input(t, label):
    if label == 'V1':
        return 1.0 * np.sign(np.sin(6.28*1e3*t))
    elif label == 'V2':
        return 2.0 * np.ones(len(t))

res = simulate(ss, t, my_input)

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.title(f"Circuit Response")
plt.plot(t*1000, res[:, ss.y_labels.index('v_2')], label='V_cap (Node 2)')
plt.ylabel('Spanning [V]'); plt.grid(True); plt.legend()

plt.subplot(212)
plt.plot(t*1000, res[:, ss.y_labels.index('i_R1')], color='orange', label='I_inductor')
plt.ylabel('Stroom [A]'); plt.xlabel('Tijd [ms]'); plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()
