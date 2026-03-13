#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:54:20 2026

@author: Marcel Hesselberth
"""

import pytest
import numpy as np
import sys, os
sys.path.append(os.getcwd() + '/..')
from parser import ExpressionParser


# ───────────────────────────────────────────────
# Fixtures & helpers
# ───────────────────────────────────────────────

@pytest.fixture
def variables():
    return {
        'Vcc': 5.0,
        'Vdd': 12.0,
        'tau': 0.01,
        'f': 50.0,
        'omega': 100.0,
        'phase': 0.0,
        'pi': np.pi,
        'e': np.e,
    }


@pytest.fixture
def parser(variables):
    return ExpressionParser(variables=variables)


@pytest.fixture
def t_small():
    return np.array([0.0, 0.002, 0.005, 0.01])


@pytest.fixture
def t_empty():
    return np.array([])


RTOL = 1e-9
ATOL = 1e-12


def assert_array_close(actual, expected):
    if np.isscalar(expected):
        expected = np.full_like(actual, expected)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


# ───────────────────────────────────────────────
# Positive tests – should succeed
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_at_0", [
    ("2 + 3", 5.0),
    ("-5 + 7.5", 2.5),
    ("2 * -3", -6.0),
    ("2 ** 3 ** 2", 512.0),
    ("- - - -8", 8.0),
    ("sin(pi/2)", 1.0),
    ("abs(-3.2)", 3.2),
    ("sqrt(16)", 4.0),
    ("ramp(-2)", 0.0),
    ("Vcc * (1 - exp(-t / tau))", 0.0),           # at t=0
    ("1.23e-4 + 5", 5.000123),
    ("-4.5e+2 * 2", -900.0),
    (".5e-1", 0.05),
    ("-sqrt(16)", -4.0),
    ("--abs(-5)", 5.0),
])
def test_correct_expressions(parser, t_small, expr, expected_at_0):
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_small)
    assert_array_close(result[0], expected_at_0)


# ───────────────────────────────────────────────
# Error cases 
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_exc_msg_substring", [
    # Mismatched parentheses
    ("(1 + 2",           "Mismatched parentheses"),
    ("1 + 2)",           "Mismatched parentheses"),
    ("((3 * 4)",         "Mismatched parentheses"),
    (") + 5",            "Mismatched parentheses"),

    # Wrong number of function arguments
    ("sin()",            "Missing argument(s) for function 'sin'"),
    ("sin(1,2)",         "Invalid expression"),
    ("pwm(t)",           "Missing argument(s) for function 'pwm'"),
    ("pwm(t, 0.5)",      "Missing argument(s) for function 'pwm'"),
    ("pulse(t,0,0.2)",   "Missing argument(s) for function 'pulse'"),

    # Syntax / missing operands
    ("3 + * 4",          "Missing left operand for '*'"),
    ("5 / / 2",          "Missing left operand for '/'"),
    ("2 3 + 4",          "Expected operator before '3'"),
    ("+ +",              "Missing operand for unary '+'"),

    # Unknown variables
    ("x + 5",            "Unknown variable: 'x'"),
    ("Vcc + unknown",    "Unknown variable: 'unknown'"),

    # Comma outside function (tends to cause stack or paren error)
    (", 5",              "Comma outside of function arguments"),
    ("5 + , 3",          "Comma outside of function arguments"),
    ("(1, 2)",           "Invalid expression"),

    # Malformed numbers — current tokenizer accepts many as variables → later fails
    ("1.2.3",            "Invalid number or unknown variable: '1.2.3'"),
    ("1e",               "Invalid number or unknown variable: '1e'"),
    ("1e+",              "Invalid number or unknown variable: '1e+'"),
    ("e4",               "Unknown variable: 'e4'"),
])
def test_error_cases_real_messages(parser, t_small, expr, expected_exc_msg_substring):
    with pytest.raises(ValueError) as exc_info:
        postfix = parser.to_postfix(expr)
        _ = parser.evaluate(postfix, t_small)

    assert expected_exc_msg_substring in str(exc_info.value)


# ───────────────────────────────────────────────
# Numerical edge cases (should not raise — produce inf/nan)
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr", [
    "1 / 0",
    "sqrt(-1)",
    "1 / (t - t)",
    "exp(1000)",
])
def test_numerical_edges_no_exception(parser, t_small, expr):
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_small)
    assert len(result) == len(t_small)
    # We allow inf / nan
    assert np.all(np.isfinite(result) | np.isnan(result) | np.isinf(result))


# ───────────────────────────────────────────────
# Empty input array
# ───────────────────────────────────────────────

def test_empty_t_array(parser, t_empty):
    expr = "Vcc * sin(2 * pi * f * t + phase)"
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_empty)
    assert result.shape == (0,)
    assert result.dtype.kind == 'f'


def test_empty_expression(parser, t_small):
    with pytest.raises(ValueError):
        # Current code returns zeros_like → but many would prefer error
        # If you want to make it raise → change evaluate when postfix empty
        result = parser.evaluate([], t_small)
        assert np.all(result == 0)


# ───────────────────────────────────────────────
# Deep nesting (should usually work until stack/recursion limit)
# ───────────────────────────────────────────────

def test_deep_nesting(parser, t_small):
    # ~20 levels — should be fine
    expr = "sin(" * 10 + "t" + ")" * 10
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_small)
    assert len(result) == len(t_small)


# ───────────────────────────────────────────────
# Prefix tests – should succeed
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_val", [
    ("1k", 1000.0),
    ("100m", 0.1),
    ("1u", 1e-6),
    ("4.7k + 300", 5000.0),
    ("Vcc * 10m", 0.05),              # Vcc (5.0) * 0.01
    ("1M / 1k", 1000.0),              # Mega / kilo
    ("sin(1000m * pi / 2)", 1.0),     # 1000m is 1
    ("500u * 2k", 1.0),               # 0.0005 * 2000
    ("1e-3 * 1k", 1.0),               # Mix van scientific en prefix
    ("100n * 10**9", 100.0),          # nano
    ("1p * 1T", 1.0),                 # pico * Tera
    ("1f * 1e15", 1.0),               # femto
    ("-10k", -10000.0),               # Unaire minus met prefix
    ("abs(-100m)", 0.1),              # Functie met prefix
    ("pwm(t, 500m, 1k)", 1.0),        # Meerdere argumenten met prefixes (bij t=0)
])
def test_prefixes_correct(parser, t_small, expr, expected_val):
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_small)
    # Check de eerste waarde (t=0)
    assert_array_close(result[0], expected_val)


@pytest.mark.parametrize("expr, expected_at_0", [
    ("1e-3", 0.001),                  # Scientific notation (geen prefix '3')
    ("1m", 0.001),                    # Prefix notation
])
def test_scientific_vs_prefix(parser, t_small, expr, expected_at_0):
    # Deze test garandeert dat 'e' in 1e3 niet als variabele of prefix wordt gezien
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_small)
    assert_array_close(result[0], expected_at_0)


# ───────────────────────────────────────────────
# Prefix Error cases
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_exc_msg_substring", [
    ("10x", "Invalid number or unknown variable: '10x'"),      # 'x' is geen prefix
    ("10 k", "Expected operator before 'k'"), # Spatie tussen getal en prefix (als k een var is)
    ("k10", "Unknown variable: 'k10'"),      # Prefix voor het getal
    ("1mk", "Invalid number or unknown variable: '1mk'"),      # Dubbele prefix (niet toegestaan)
    ("1e-3m", "Invalid number or unknown variable: '1e-3m'"),  # Prefix direct na exponent zonder spatie
    ("1.0e3k", "Invalid number or unknown variable: '1.0e3k'"),  # Prefix direct na exponen
])
def test_prefix_errors(parser, t_small, expr, expected_exc_msg_substring):
    with pytest.raises(ValueError) as exc_info:
        postfix = parser.to_postfix(expr)
        _ = parser.evaluate(postfix, t_small)
    assert expected_exc_msg_substring in str(exc_info.value)


# ───────────────────────────────────────────────
# Extra complexe expressies (Prefix + Functies)
# ───────────────────────────────────────────────

def test_complex_prefix_expression(parser, t_small):
    # Expressie: Vcc (5) * sin(2 * pi * 50 * t) + 100m
    # Bij t=0: 5 * sin(0) + 0.1 = 0.1
    expr = "Vcc * sin(2 * pi * 50 * t) + 100m"
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t_small)
    assert_array_close(result[0], 0.1)

def test_prefix_in_pwm(parser):
    # t=0, duty=0.5, freq=1kHz. PWM bij t=0 is altijd 1.0 (of 0.999...)
    t = np.array([0.0, 0.0001]) 
    expr = "pwm(t, 500m, 1k)" 
    postfix = parser.to_postfix(expr)
    result = parser.evaluate(postfix, t)
    assert result[0] == 1.0

