from phybench.evaluation.expression_distance import EED
from phybench.settings import EvaluationEEDSettings


def test_gt_with_imaginary_unit_is_supported():
    settings = EvaluationEEDSettings()
    gt = r"$$\sqrt{-\frac{2\sigma_0 q}{\varepsilon_0 \alpha^2 \left(a+\frac{1}{\alpha}\right)^3 m}}$$"
    # Compare GT to itself; should be exactly equal when parsed
    score, rel_dist, tree_size, distance = EED(gt, gt, settings)
    assert score == 100
    assert distance == 0
    assert rel_dist == 0
    # tree_size can be 0 (when built) or -1 when equal short-circuit triggers before tree build
    assert tree_size in (0, -1)
