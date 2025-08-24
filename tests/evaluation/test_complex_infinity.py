import sympy as sp
from phybench.evaluation.expression_distance import EED, NodeType, sympy_to_tree
from phybench.settings import EvaluationEEDSettings


def test_complex_infinity_constant_supported():
    settings = EvaluationEEDSettings()
    # A LaTeX expression that evaluates to complex infinity in SymPy context is tricky; instead,
    # ensure we can compare an expression containing division by zero, which sympy treats via zoo.
    # We compare identical forms so the pipeline should short-circuit to equality.
    gt = r"$\frac{1}{0}$"
    score, rel_dist, tree_size, distance = EED(gt, gt, settings)
    assert score == 100
    assert distance == 0
    assert rel_dist == 0
    assert tree_size in (0, -1)


def test_sympy_to_tree_uses_complex_infinity_node():
    # Directly verify tree conversion labels ComplexInfinity as a number node
    node = sympy_to_tree(sp.zoo)
    assert node.label == f"{NodeType.NUMBER.value}_{sp.zoo}"


def test_sympy_to_tree_with_expression_containing_zoo():
    # Build a composite expression that contains zoo to force recursion
    x = sp.Symbol("x")
    expr = sp.zoo + x
    tree = sympy_to_tree(expr)

    def contains_zoo(n):
        if n.label == f"{NodeType.NUMBER.value}_{sp.zoo}":
            return True
        return any(contains_zoo(c) for c in n.children)

    assert contains_zoo(tree)
