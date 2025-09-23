import numpy as np
import torch


def test_critic_weights_tensor_and_invariants():
    from src.components.scorer.critic_delta import CriticDeltaScorer

    scorer = CriticDeltaScorer({"temperature": 0.7})
    scorer.setup({})
    out = scorer.run({"seq_lengths": [16], "rewards": [1.0]})

    # tensor 반환 확인
    assert "weights" in out
    w_t = out["weights"]
    assert isinstance(w_t, torch.Tensor)
    assert torch.isfinite(w_t).all()

    # 통계 불변식
    w = w_t.detach().cpu().numpy()  # tensor->np
    m = float(np.mean(w))
    assert 0.95 <= m <= 1.05
    assert np.all(w > 0)


def test_rho1_weights_tensor_and_invariants():
    from src.components.scorer.rho1_excess import Rho1ExcessScorer

    scorer = Rho1ExcessScorer({"temperature": 0.7})
    scorer.setup({})
    out = scorer.run({"seq_lengths": [16]})

    assert "weights" in out
    w_t = out["weights"]
    assert isinstance(w_t, torch.Tensor)
    assert torch.isfinite(w_t).all()

    w = w_t.detach().cpu().numpy()  # tensor->np
    m = float(np.mean(w))
    assert 0.95 <= m <= 1.05
    assert np.all(w > 0)


