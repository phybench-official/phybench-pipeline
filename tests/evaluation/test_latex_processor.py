from phybench.evaluation.latex_processor import master_convert


def test_simple_latex_processing():
    """Tests basic LaTeX processing."""
    latex_string = "x + y"
    sympy_expr = master_convert(latex_string)
    assert str(sympy_expr) == "x + y"


# def test_complex_fraction_processing():
#     """Tests processing of a complex LaTeX fraction."""
#     latex_string = "\frac{a}{b+c}"
#     sympy_expr = master_convert(latex_string)
#     assert str(sympy_expr) == "a/(b + c)"


def test_invalid_latex_string():
    """Tests handling of invalid LaTeX strings."""
    latex_string = "\frac{a}{b+c"  # Missing closing brace
    sympy_expr = master_convert(latex_string)
    assert sympy_expr is None
