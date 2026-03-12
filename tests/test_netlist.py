#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 00:04:57 2026

@author: hessel
"""

import pytest
import numpy as np
import math
import re
from lcapy import Circuit
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from netlist3 import Netlist, ExpressionParser

@pytest.fixture
def empty_netlist():
    return Netlist("")

@pytest.fixture
def simple_netlist():
    return Netlist("""
    R1 1 0 100
    V1 1 0 10
    """)

@pytest.fixture
def parser():
    return ExpressionParser()

# Tests for ExpressionParser
def test_empty_expression(parser):
    postfix = parser.to_postfix("")
    assert np.allclose(parser.evaluate(postfix, np.array([0])), 0)

def test_invalid_parens(parser):
    with pytest.raises(ValueError):  # or specific error
        parser.to_postfix("sin(t")

def test_operator_without_operands(parser):
    postfix = parser.to_postfix("+")
    with pytest.raises(ValueError):
        parser.evaluate(postfix, np.array([0]))

def test_unknown_function(parser):
    with pytest.raises(KeyError):
        parser.to_postfix("foo(t)")

def test_function_missing_args(parser):
    postfix = parser.to_postfix("sin()")
    with pytest.raises(ValueError):
        parser.evaluate(postfix, np.array([0]))

def test_div_by_zero(parser):
    postfix = parser.to_postfix("1/0")
    with pytest.raises(ZeroDivisionError):
        parser.evaluate(postfix, np.array([0]))

def test_negative_value(parser):
    postfix = parser.to_postfix("-5")
    assert np.allclose(parser.evaluate(postfix, np.array([0])), -5)

def test_complex_expression(parser):
    postfix = parser.to_postfix("sin(2*pi*t) + 1")
    t = np.array([0, 0.25])
    expected = np.sin(2*np.pi*t) + 1
    assert np.allclose(parser.evaluate(postfix, t), expected)

# Tests for Netlist
def test_empty_netlist(empty_netlist):
    assert len(empty_netlist.components) == 0
    assert len(empty_netlist.variables) == 0
    assert empty_netlist.get_sim_netlist() == ""
    assert empty_netlist.get_draw_netlist() == ""
    with pytest.raises(KeyError):
        empty_netlist.get_signal_func("V1")

def test_only_comments():
    nl = Netlist("""
    # comment
    """)
    assert len(nl.components) == 0

def test_standalone_meta():
    nl = Netlist("""
    ; style=american
    """)
    assert nl.get_draw_netlist() == "; style=american"

def test_wrong_number_of_nodes():
    nl = Netlist("""
    R1 1
    """)
    assert len(nl.components) == 0

def test_resistor_with_esr_like():
    nl = Netlist("""
    R1 1 0 100 0.1
    """)
    assert isinstance(nl.components[0]['val'], str)
    assert nl.components[0]['esr'] == 0.0
    assert "100 0.1" in nl.get_sim_netlist()

    # Check if Circuit fails
    with pytest.raises(ValueError):
        clean_lines = [l for l in nl.get_draw_netlist().split('\n') if '=' not in l.split(';')[0]]
        Circuit('\n'.join(clean_lines))

def test_cap_with_invalid_esr():
    nl = Netlist("""
    C1 1 0 1u foo
    """)
    assert nl.components[0]['val'] == 1e-6
    assert nl.components[0]['esr'] == "foo"
    assert "foo" in nl.get_sim_netlist()

def test_voltage_invalid_function():
    nl = Netlist("""
    V1 1 0 sin(t
    """)
    with pytest.raises(ValueError):
        nl.get_signal_func("V1")(np.array([0]))

def test_operator_no_args():
    nl = Netlist("""
    V1 1 0 +
    """)
    with pytest.raises(ValueError):
        nl.get_signal_func("V1")(np.array([0]))

def test_duplicate_names():
    nl = Netlist("""
    R1 1 0 100
    R1 2 0 200
    """)
    assert len(nl.components) == 2
    # gets first
    assert nl.get_signal_func("R1")(np.array([0])) == 100

def test_gnd_no_node():
    nl = Netlist("gnd")
    assert len(nl.components) == 0

def test_gnd_with_node():
    nl = Netlist("gnd 1")
    assert len(nl.components) == 1
    assert "ground" in nl.components[0]['meta']

def test_invalid_variable():
    nl = Netlist("a = b")
    assert nl.variables['a'] == "b"

def test_pretty_prefix():
    nl = Netlist("")
    assert nl._pretty_prefix(1000) == "1k"
    assert nl._pretty_prefix(1e-6) == "1u"
    assert nl._pretty_prefix(0) == "0"
    assert nl._pretty_prefix("str") == "str"

def test_valid_function():
    nl = Netlist("""
    V1 1 0 sin(2*pi*t)
    """)
    func = nl.get_signal_func("V1")
    t = np.array([0, 0.25])
    assert np.allclose(func(t), [0, 1])

def test_draw_valid(simple_netlist):
    fig, ax = plt.subplots()
    simple_netlist.draw_to_axis(ax)

def test_draw_invalid():
    nl = Netlist("""
    R1 1 0 foo
    """)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        nl.draw_to_axis(ax)

def test_switch():
    nl = Netlist("""
    SW1 1 0
    """)
    assert "{R_SW1}" in nl.get_sim_netlist()

def test_diode():
    nl = Netlist("""
    D1 1 0
    """)
    sim = nl.get_sim_netlist()
    assert "0.7" in sim
    assert "10m" in sim

# Additional tests can be added for more coverage.