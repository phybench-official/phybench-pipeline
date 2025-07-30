from phybench.evaluation.expression_distance import EED
from phybench.settings import EvaluationEEDSettings

eed_settings = EvaluationEEDSettings()

# --- Test Data ---
ground_truth_latex_332 = r"""
\mu = \frac{MR^2 \sin\varphi \sin\left(\arccos\frac{L^2 + d^2 - R^2}{2L d} + \arccos\frac{R^2 + d^2 - L^2}{2R d}\right)}{\sqrt{\left(\frac{L^2 + d^2 - R^2}{2 L d}\right)^2 \left[ M R^2 \cos\varphi + m L^2 \sin^2\left(\arccos\frac{L^2 + d^2 - R^2}{2L d}\right) \left(\frac{3}{2} \cos\varphi - 1\right)\right]^2 + \left[ M R^2 \sin\varphi \cos\left(\arccos\frac{L^2 + d^2 - R^2}{2L d} + \arccos\frac{R^2 + d^2 - L^2}{2R d}\right)\right]^2 }}
"""
model_latex_332 = r"\frac{(d \cos \varphi - \sqrt{d^2 - R^2 - L^2 + 2 d L \cos \varphi}) \sin \varphi}{R + \sqrt{d^2 - R^2 - L^2 + 2 d L \cos \varphi} \cos \varphi}"

ground_truth_latex_698 = r"""
\bar{\eta}_k(p) = \frac{(p+4\nu k^2)\eta_k(0)+\eta'_k(0)}{(p+2\nu k^2)^2+(g+\frac{\tau}{\rho} k^2)k \tanh(kh) - 4\nu^{3/2}k^3\frac{\sqrt{p+\nu k^2}\tanh(kh)}{\tanh\left(\sqrt{\frac{p+\nu k^2}{\nu}}h\right)}}
"""
model_latex_698 = r"\bar{\eta}_k(p) = \frac{\eta_k(0)(p + 2\nu k^2) + \eta'_k(0)}{p^2 + 2\nu k^2 p + \frac{\tau k^3 \tanh(kh)}{\rho}}"

ground_truth_latex_713 = r"""
\frac{\sqrt{1-\beta^2}}{4\pi}\frac{\frac{\sqrt{1-\beta^2}E_1\beta\cos\theta+\sqrt{\left(1-\beta^2\right)E_1^2-\left(1-\beta^2\cos^2\theta\right)m_\tau^2c^4}}{1-\beta^2\cos^2\theta}}{\sqrt{E_1^2-m_\tau^2c^4}\left(1-\frac{\beta\cos\theta\sqrt{\left(\frac{\sqrt{1-\beta^2}E_1\beta\cos\theta+\sqrt{\left(1-\beta^2\right)E_1^2-\left(1-\beta^2\cos^2\theta\right)m_\tau^2c^4}}{1-\beta^2\cos^2\theta}\right)^2+m_\tau^2c^4}}{\frac{\sqrt{1-\beta^2}E_1\beta\cos\theta+\sqrt{\left(1-\beta^2\right)E_1^2-\left(1-\beta^2\cos^2\theta\right)m_\tau^2c^4}}{1-\beta^2\cos^2\theta}}\right)}
"""
model_latex_713 = r"\frac{(1 - \beta^2) E_1^2}{(E_1 - \beta \sqrt{E_1^2 - m_\tau^2 c^4} \cos\theta)^2 \sqrt{1 - \beta^2 \cos^2\theta}}"


def test_issue_332_regression():
    """Tests for regression on issue 332."""
    eed_settings_332 = EvaluationEEDSettings(simplify_time_limit=300)
    score, _, ans_size, distance = EED(
        answer_latex=ground_truth_latex_332,
        test_latex=model_latex_332,
        eed_settings=eed_settings_332,
    )
    assert ans_size == 159
    assert round(distance, 1) == 110.4


def test_issue_698_regression():
    """Tests for regression on issue 698."""
    score, _, ans_size, distance = EED(
        answer_latex=ground_truth_latex_698,
        test_latex=model_latex_698,
        eed_settings=eed_settings,
    )
    assert ans_size == 110
    assert distance == 58.4


def test_issue_713_regression():
    """Tests for regression on issue 713."""
    score, _, ans_size, distance = EED(
        answer_latex=ground_truth_latex_713,
        test_latex=model_latex_713,
        eed_settings=eed_settings,
    )
    assert ans_size == 234
    assert distance == 141.4
