import pytest

torch = pytest.importorskip("torch")

from model import GPT, ByteTokenizer, GPTConfig  # noqa: E402


def tiny():
    return GPT(GPTConfig(context=32, dim=64, n_layers=2, n_heads=4, n_kv_heads=2))


def test_forward_shapes_and_loss():
    m = tiny()
    x, y = torch.randint(0, 256, (2, 16)), torch.randint(0, 256, (2, 16))
    logits, loss = m(x, y)
    assert logits.shape == (2, 16, 256)
    assert abs(loss.item() - torch.log(torch.tensor(256.0)).item()) < 0.5


def test_causality():
    m = tiny().eval()
    x = torch.randint(0, 256, (1, 16))
    y = x.clone(); y[0, 10] = (y[0, 10] + 1) % 256
    a, _ = m(x); b, _ = m(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)


def test_learns_to_overfit():
    torch.manual_seed(0)
    m = tiny()
    tok = ByteTokenizer()
    data = torch.tensor([tok.encode("hello world! " * 2)])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = m(data[:, :-1], data[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.3


def test_generate_respects_should_stop():
    m = tiny().eval()
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 5

    out = m.generate(torch.zeros(1, 1, dtype=torch.long), 100, should_stop=stop)
    assert out.shape[1] == 1 + 5
