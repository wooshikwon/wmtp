import torch


def make_logits_labels(B=1, S=6, H=3, V=10):
    torch.manual_seed(0)
    logits = torch.randn(B, S, H, V)
    labels = torch.randint(low=0, high=V, size=(B, S))
    return logits, labels


def naive_mtp_ce_avg(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    B, S, H, V = logits.shape
    ce_sum = torch.zeros(B, S)
    count = torch.zeros(B, S)
    for k in range(H):
        shift = k + 1
        valid_len = S - shift
        if valid_len <= 0:
            break
        l_k = logits[:, :valid_len, k, :].transpose(1, 2)  # [B,V,valid]
        y_k = labels[:, shift:shift + valid_len]
        ce_k = torch.nn.functional.cross_entropy(l_k, y_k, reduction="none", ignore_index=ignore_index)
        ce_sum[:, :valid_len] += ce_k
        valid_mask = (y_k != ignore_index).to(ce_sum.dtype)
        count[:, :valid_len] += valid_mask
    count = torch.clamp(count, min=1.0)
    return ce_sum / count


def test_compute_mtp_ce_alignment_matches_naive():
    # Import target helper
    from src.components.trainer.mtp_weighted_ce_trainer import _compute_mtp_ce_loss

    logits, labels = make_logits_labels()
    ce_avg_ref = naive_mtp_ce_avg(logits, labels)
    ce_avg_impl, valid_mask = _compute_mtp_ce_loss(logits, labels, horizon=3, ignore_index=-100)

    # Only compare where valid_mask is True
    diff = (ce_avg_ref[valid_mask] - ce_avg_impl[valid_mask]).abs().max().item() if valid_mask.any() else 0.0
    assert diff < 1e-6


