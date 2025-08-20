import phybench.evaluation.expression_distance as eedmod
import pytest
from loguru import logger
from phybench.evaluation.expression_distance import EED
from phybench.settings import EvaluationEEDSettings

eed_settings = EvaluationEEDSettings()


def test_timeout_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """EED should log timeouts and fall back gracefully when operations exceed limits.

    This test verifies intended behavior by:
    - Forcing simplify/equality timeouts via tiny limits.
    - Capturing loguru warnings emitted on timeouts.
    - Monkeypatching heavy steps (expand, ext_distance) to keep runtime minimal.
    """

    # Complex expressions (content not essential once we stub heavy steps)
    ground_truth_latex = r"""
\mu = \frac{MR^2 \sin\varphi \sin\left(\arccos\frac{L^2 + d^2 - R^2}{2L d} + \arccos\frac{R^2 + d^2 - L^2}{2R d}\right)}{\sqrt{\left(\frac{L^2 + d^2 - R^2}{2 L d}\right)^2 \left[ M R^2 \cos\varphi + m L^2 \sin^2\left(\arccos\frac{L^2 + d^2 - R^2}{2L d}\right) \left(\frac{3}{2} \cos\varphi - 1\right)\right]^2 + \left[ M R^2 \sin\varphi \cos\left(\arccos\frac{L^2 + d^2 - R^2}{2L d} + \arccos\frac{R^2 + d^2 - L^2}{2R d}\right)\right]^2 }}
"""
    model_latex = r"\frac{(d \cos \varphi - \sqrt{d^2 - R^2 - L^2 + 2 d L \cos \varphi}) \sin \varphi}{R + \sqrt{d^2 - R^2 - L^2 + 2 d L \cos \varphi} \cos \varphi}"

    # Force timeouts (values aren't critical because we monkeypatch the calls to raise)
    eed_settings_timeout = EvaluationEEDSettings(
        simplify_time_limit=1, equals_time_limit=1
    )

    # Stub heavy operations: expand (pre-simplify) and tree distance
    monkeypatch.setattr(
        "phybench.evaluation.expression_distance.expand",
        lambda expr: expr,
        raising=True,
    )
    stub_distance = 123.45
    monkeypatch.setattr(
        "phybench.evaluation.expression_distance.ext_distance",
        lambda *a, **k: stub_distance,
        raising=True,
    )

    # Force simplify/equality paths to raise timeouts so we can assert logs deterministically
    monkeypatch.setattr(
        "phybench.evaluation.expression_distance.simplify_with_timeout",
        lambda *a, **k: (_ for _ in ()).throw(
            eedmod.TimeoutError("forced simplify timeout")
        ),
        raising=True,
    )
    monkeypatch.setattr(
        "phybench.evaluation.expression_distance.equal_with_timeout",
        lambda *a, **k: (_ for _ in ()).throw(
            eedmod.TimeoutError("forced equals timeout")
        ),
        raising=True,
    )

    # Capture timeout warnings from loguru
    logs: list[str] = []
    sink_id = logger.add(lambda m: logs.append(str(m)), level="WARNING")
    try:
        score, _, ans_size, distance = EED(
            answer_latex=ground_truth_latex,
            test_latex=model_latex,
            eed_settings=eed_settings_timeout,
        )
    finally:
        logger.remove(sink_id)

    # Verify intended behavior
    assert any("Simplification timed out" in m for m in logs)
    assert any("Equality check timed out" in m for m in logs)
    # Ensure we proceeded to distance calculation (stub used) without crashing
    assert distance == stub_distance
