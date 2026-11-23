import numpy as np
import pytest
import torch

from rul.engine import DeviceBatcher, nasa_score, regression_metrics, train_model
from rul.model import LSTMRegressor, RULEnsemble, RULTransformer, load_checkpoint, save_checkpoint


@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_one_prediction_per_sequence(pooling):
    m = RULTransformer(n_features=7, d_model=16, nhead=4, num_layers=1, pooling=pooling).eval()
    assert m(torch.randn(5, 30, 7)).shape == (5,)


def test_lstm_baseline_one_prediction_per_sequence():
    m = LSTMRegressor(n_features=7, hidden_size=16, num_layers=2).eval()
    assert m(torch.randn(5, 30, 7)).shape == (5,)


def test_lstm_stack_matches_builtin_multilayer_lstm():
    """Our stack of single-layer LSTMs computes exactly what nn.LSTM(num_layers=2) does."""
    torch.manual_seed(0)
    ours = LSTMRegressor(n_features=5, hidden_size=8, num_layers=2, dropout=0.3).eval()
    ref = torch.nn.LSTM(5, 8, num_layers=2, batch_first=True).eval()
    with torch.no_grad():
        for layer, lstm in enumerate(ours.layers):
            for name in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
                getattr(ref, f"{name}_l{layer}").copy_(getattr(lstm, f"{name}_l0"))
    x = torch.randn(4, 12, 5)
    stacked = x
    for i, lstm in enumerate(ours.layers):
        stacked, _ = lstm(stacked if i == 0 else ours.between(stacked))
    torch.testing.assert_close(stacked, ref(x)[0])


def test_positional_encoding_makes_order_matter():
    torch.manual_seed(0)
    m = RULTransformer(n_features=3, d_model=16, nhead=2, num_layers=1, dropout=0.0).eval()
    x = torch.randn(1, 10, 3)
    assert not torch.allclose(m(x), m(x.flip(1)))


def test_invalid_heads_rejected():
    with pytest.raises(ValueError):
        RULTransformer(n_features=3, d_model=10, nhead=4)


def test_checkpoint_roundtrip_single_and_ensemble(tmp_path):
    torch.manual_seed(0)
    members = [RULTransformer(n_features=4, d_model=16, nhead=2, num_layers=1).eval() for _ in range(3)]
    x = torch.randn(8, 12, 4)
    expected = RULEnsemble(members).eval()(x)
    save_checkpoint(tmp_path / "m.pt", members, seq_len=12)
    model, ckpt = load_checkpoint(tmp_path / "m.pt")
    assert isinstance(model, RULEnsemble) and ckpt["seq_len"] == 12
    torch.testing.assert_close(model(x), expected)

    save_checkpoint(tmp_path / "s.pt", members[:1])
    single, _ = load_checkpoint(tmp_path / "s.pt")
    torch.testing.assert_close(single(x), members[0](x))


def test_nasa_score_is_asymmetric():
    assert nasa_score([10], [10]) == 0
    late, early = nasa_score([50], [60]), nasa_score([50], [40])
    assert late == pytest.approx(np.e - 1) and early == pytest.approx(np.exp(10 / 13) - 1)
    assert late > early
    m = regression_metrics([0, 0], [3, -4])
    assert m["rmse"] == pytest.approx(np.sqrt(12.5)) and m["mae"] == pytest.approx(3.5)


def test_device_batcher_covers_every_sample_once():
    X = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    b = DeviceBatcher(X, np.arange(10), batch_size=4, device="cpu", shuffle=True, seed=1)
    seen = torch.cat([y for _, y in b]).tolist()
    assert len(b) == 3 and sorted(seen) == list(range(10))


def test_training_learns_and_restores_best_epoch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 8, 2)).astype(np.float32)
    y = (50 + 20 * X[:, -1, 0]).astype(np.float32)  # learnable target
    data = dict(X_train=X[:200], y_train=y[:200], X_val=X[200:], y_val=y[200:])
    kw = dict(n_features=2, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0)
    model, hist, best_epoch, best_val = train_model(data, kw, lr=3e-3, batch_size=32, epochs=15, patience=15,
                                                    weight_decay=0.0, loss="mse", device="cpu", seed=0,
                                                    target_scale=100.0, verbose=False)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert best_val == min(h["val_loss"] for h in hist) == hist[best_epoch - 1]["val_loss"]
    assert not model.training
