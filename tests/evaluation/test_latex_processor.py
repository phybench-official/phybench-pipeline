import pytest
from phybench.evaluation.latex_processor import master_convert


def test_simple_latex_processing():
    """Tests basic LaTeX processing."""
    latex_string = r"x + y"
    sympy_expr = master_convert(latex_string)
    assert str(sympy_expr) == "x + y"


def test_complex_fraction_processing():
    """Tests processing of a complex LaTeX fraction."""
    latex_string = r"\frac{a}{b+c}"
    sympy_expr = master_convert(latex_string)
    assert str(sympy_expr) == "a/(b + c)"


def test_invalid_latex_string():
    """Tests handling of invalid LaTeX strings."""
    latex_string = r"\frac{a}{b+c"  # Missing closing brace
    # We are catching a generic Exception here because the underlying
    # latex2sympy2 library raises a generic Exception.
    with pytest.raises(Exception):  # noqa: B017
        master_convert(latex_string)


def test_trigonometric_functions():
    """Tests processing of trigonometric functions."""
    latex_string = r"\sin(x) + \cos(y)"
    sympy_expr = master_convert(latex_string)
    assert str(sympy_expr) == "sin(x) + cos(y)"


def test_subscripts_and_superscripts():
    """Tests processing of subscripts and superscripts."""
    latex_string = r"x_1^2 + y_2^3"
    sympy_expr = master_convert(latex_string)
    assert str(sympy_expr) == "x_1**2 + y_2**3"


def test_greek_letters():
    """Tests processing of Greek letters."""
    latex_string = r"\alpha + \beta"
    sympy_expr = master_convert(latex_string)
    assert str(sympy_expr) == "alpha + beta"
