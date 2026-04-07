"""Unit tests for the safe arithmetic evaluator in general_tools.

These tests run without a live MCP server. They exercise the
_safe_eval helper directly and document both valid inputs and
injection/DoS attempts that must be rejected.
"""

import pytest

from agentic_framework_pkg.mcp_server.tools.general_tools import _safe_eval


class TestSafeEvalValid:
    """Valid arithmetic expressions should return the correct numeric result."""

    def test_addition(self):
        assert _safe_eval("2 + 3") == 5

    def test_subtraction(self):
        assert _safe_eval("10 - 4") == 6

    def test_multiplication(self):
        assert _safe_eval("3 * 7") == 21

    def test_division(self):
        assert _safe_eval("10 / 4") == pytest.approx(2.5)

    def test_parentheses(self):
        assert _safe_eval("(2 + 3) * 4") == 20

    def test_nested_parentheses(self):
        assert _safe_eval("((1 + 2) * (3 + 4))") == 21

    def test_float_arithmetic(self):
        assert _safe_eval("1.5 + 2.5") == pytest.approx(4.0)

    def test_unary_minus(self):
        assert _safe_eval("-3 + 5") == 2

    def test_integer_result(self):
        assert _safe_eval("100 / 4") == pytest.approx(25.0)

    def test_order_of_operations(self):
        assert _safe_eval("2 + 3 * 4") == 14


class TestSafeEvalRejection:
    """Dangerous or unsupported inputs must raise ValueError, not execute."""

    def test_reject_arbitrary_name(self):
        """Names (variables, builtins) must not be resolved."""
        with pytest.raises((ValueError, TypeError)):
            _safe_eval("os.system('ls')")

    def test_reject_function_call(self):
        """Function calls must be rejected."""
        with pytest.raises((ValueError, TypeError)):
            _safe_eval("__import__('os').system('ls')")

    def test_reject_string_literal(self):
        """Non-numeric literals must be rejected."""
        with pytest.raises((ValueError, TypeError)):
            _safe_eval("'hello'")

    def test_reject_exponentiation_dos(self):
        """Gigantic exponent that would hang a naive eval must be rejected or
        complete quickly (guard: exponent limit)."""
        with pytest.raises((ValueError, OverflowError)):
            _safe_eval("9 ** 9 ** 9")

    def test_reject_empty_string(self):
        with pytest.raises((ValueError, SyntaxError)):
            _safe_eval("")

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            _safe_eval("1 / 0")
