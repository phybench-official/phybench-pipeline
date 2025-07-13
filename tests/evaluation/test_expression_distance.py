from phybench.evaluation.expression_distance import EED


def test_identical_expressions():
    """Tests that identical expressions have a distance of 0."""
    latex1 = "a + b"
    latex2 = "a + b"
    score, _, _, distance = EED(latex1, latex2, scoring_parameters=[60, 100])
    assert score == 100
    assert distance == 0


def test_slightly_different_expressions():
    """Tests the distance of slightly different expressions."""
    latex1 = "a + b"
    latex2 = "a - b"
    score, _, _, distance = EED(latex1, latex2, scoring_parameters=[60, 100])
    assert score < 100
    assert distance > 0


def test_structurally_different_expressions():
    """Tests the distance of structurally different expressions."""
    latex1 = "\\frac{a}{b}"
    latex2 = "a + b"
    score, _, _, distance = EED(latex1, latex2, scoring_parameters=[60, 100])
    assert score < 100
    assert distance > 0


# def test_real_world_example_from_results():
#     """Tests a real-world example from the evaluation results."""
#     ground_truth_latex = "$v = \frac{N Z e^3}{2 c \varepsilon_0 m^2 n} \frac{\omega^2}{(\omega_0^2 - \omega^2)^2}$"
#     model_answer_latex = "v = \frac{e^2 N Z}{2 \varepsilon_0 m_e c (\omega^2 - \omega_0^2)}"
#
#     score, _, _, distance = EED(ground_truth_latex, model_answer_latex, scoring_parameters=[60, 100])
#
#     # Based on the provided JSON, the distance is 9.0.
#     # The score is not directly asserted as it depends on the scoring logic which is not the focus here.
#     assert distance == 9.0
