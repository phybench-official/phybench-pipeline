from phybench.evaluation.expression_distance import EED
from phybench.settings import EvaluationEEDSettings

eed_settings = EvaluationEEDSettings()


def test_timeout_handling():
    """Tests that the EED function correctly handles timeouts."""

    # This is a complex expression that is likely to time out with a short timeout.
    ground_truth_latex = r"""
\mu = \frac{MR^2 \sin\varphi \sin\left(\arccos\frac{L^2 + d^2 - R^2}{2L d} + \arccos\frac{R^2 + d^2 - L^2}{2R d}\right)}{\sqrt{\left(\frac{L^2 + d^2 - R^2}{2 L d}\right)^2 \left[ M R^2 \cos\varphi + m L^2 \sin^2\left(\arccos\frac{L^2 + d^2 - R^2}{2L d}\right) \left(\frac{3}{2} \cos\varphi - 1\right)\right]^2 + \left[ M R^2 \sin\varphi \cos\left(\arccos\frac{L^2 + d^2 - R^2}{2L d} + \arccos\frac{R^2 + d^2 - L^2}{2R d}\right)\right]^2 }}
"""
    model_latex = r"\frac{(d \cos \varphi - \sqrt{d^2 - R^2 - L^2 + 2 d L \cos \varphi}) \sin \varphi}{R + \sqrt{d^2 - R^2 - L^2 + 2 d L \cos \varphi} \cos \varphi}"

    # Test with a very short timeout to ensure it times out.
    eed_settings_timeout = EvaluationEEDSettings(simplify_time_limit=1)
    score, _, ans_size, distance = EED(
        answer_latex=ground_truth_latex,
        test_latex=model_latex,
        eed_settings=eed_settings_timeout,
    )

    # When a timeout occurs, the answer size should be larger than the simplified size.
    assert ans_size > 159
    assert distance > 110.4
