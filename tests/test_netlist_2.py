#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:02:32 2026

@author: hessel
"""

import pytest
import numpy as np
from netlist import Netlist, ExpressionParser


def test_empty_netlist():
    """Tests if an empty string or whitespace crashes the parser."""
    nl = Netlist("")
    assert len(nl.components) == 0
    assert nl.get_sim_netlist() == ""

def test_missing_nodes():
    """Now checks that the parser DOES NOT crash, but skips invalid lines."""
    # This line has only 'R1 NodeA', missing NodeB and Value.
    nl = Netlist("R1 NodeA") 
    assert len(nl.components) == 0 # It should be skipped by the 'if len(parts) < 3' check

def test_meta_only_lines():
    """Tests if lines starting with metadata (;) are handled."""
    nl = Netlist("; standalone comment\n# another comment")
    assert len(nl.standalone_meta) == 1
    assert "standalone comment" in nl.standalone_meta

def test_function_component_collision():
    """Tests if a component name matching a function causes issues."""
    # 'sin' is a function, but 'sin' here is a node name.
    nl = Netlist("R1 sin 0 10k")
    assert "sin" in nl.node_map
    assert nl.node_map["sin"] == 1

def test_malformed_formula():
    """Tests if an invalid math expression raises the correct error."""
    parser = ExpressionParser()
    t = np.array([0, 1, 2])
    # Case: Missing operand (1 +)
    postfix = parser.to_postfix("1 +")
    
    # Change IndexError to ValueError to match your manual check
    with pytest.raises(ValueError, match="Missing operands"):
        parser.evaluate(postfix, t)
        
def test_unknown_variable_defaults():
    """Tests if an undefined variable defaults to 0.0 without crashing."""
    parser = ExpressionParser(variables={})
    postfix = parser.to_postfix("unknown_var + 1")
    t = np.array([0, 1, 2]) # Fix: Pass a valid list to np.array
    result = parser.evaluate(postfix, t)
    # 0.0 (default) + 1.0 = 1.0
    assert np.all(result == 1.0)

def test_prefix_parsing_failure():
    """Tests if invalid engineering notation (e.g. 10Z) crashes."""
    nl = Netlist("R1 1 0 10Z")
    # Current _interpret_value returns the string "10Z" if float conversion fails
    assert nl.components[0]['val'] == "10Z"