from phybench.evaluation.expression_distance import EED
from phybench.settings import EvaluationEEDSettings

eed_settings = EvaluationEEDSettings()


def test_identical_expressions():
    """Tests that identical expressions have a distance of 0."""
    latex1 = r"a + b"
    latex2 = r"a + b"
    score, _, _, distance = EED(latex1, latex2, eed_settings)
    assert score == 100
    assert distance == 0


def test_slightly_different_expressions():
    """Tests the distance of slightly different expressions."""
    latex1 = r"a + b"
    latex2 = r"a - b"
    score, _, _, distance = EED(latex1, latex2, eed_settings)
    assert score < 100
    assert distance > 0


def test_structurally_different_expressions():
    """Tests the distance of structurally different expressions."""
    latex1 = r"\frac{a}{b}"
    latex2 = r"a + b"
    score, _, _, distance = EED(latex1, latex2, eed_settings)
    assert score < 100
    assert distance > 0


def test_real_world_example_from_results():
    """Tests a real-world example from the evaluation results."""
    ground_truth_latex = r"$v = \frac{N Z e^3}{2 c \varepsilon_0 m^2 n} \frac{\omega^2}{(\omega_0^2 - \omega^2)^2}$"
    model_answer_latex = (
        r"v = \frac{e^2 N Z}{2 \varepsilon_0 m_e c (\omega^2 - \omega_0^2)}"
    )

    score, _, _, distance = EED(ground_truth_latex, model_answer_latex, eed_settings)

    # Based on the provided JSON, the distance is 9.0.
    # The score is not directly asserted as it depends on the scoring logic which is not the focus here.
    assert distance == 9.0


def test_equivalent_but_different_notation():
    """Tests expressions that are mathematically equivalent but use different notation."""
    latex1 = r"a*b"
    latex2 = r"ab"
    score, _, _, distance = EED(latex1, latex2, eed_settings)
    assert score == 100
    assert distance == 0


def test_with_constants():
    """Tests expressions with numeric constants."""
    latex1 = r"2*x + 1"
    latex2 = r"2*x + 2"
    score, _, _, distance = EED(latex1, latex2, eed_settings)
    assert score < 100
    assert distance > 0
