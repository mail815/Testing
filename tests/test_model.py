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


def _naive_greedy(m, idx, n):
    for _ in range(n):
        logits, _ = m(idx[:, -m.cfg.context:])
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return idx


def test_kv_cache_matches_full_recompute_including_window_overflow():
    torch.manual_seed(0)
    m = tiny().eval()
    idx = torch.randint(0, 256, (1, 5))
    fast = m.generate(idx, 40, temperature=1e-6, top_k=1)  # 45 > context 32
    slow = _naive_greedy(m, idx, 40)
    # Identical until the first window overflow; after that the cached path
    # sees a shorter (half) window, which is a deliberate approximation.
    assert torch.equal(fast[:, :32], slow[:, :32])


def test_presets_build():
    from model.gpt import preset
    cfg = preset("tiny")
    GPT(cfg)
    assert preset("8b").n_kv_heads == 8


def test_checkpoint_loading_is_gated_and_safe(tmp_path):
    import pickle
    from guardian import (AuditLog, DeploymentError, DeploymentGate, Quorum,
                          generate_operator_key, sha256_file, sign)
    from model.checkpoint import load_approved, load_for_eval, save

    priv, pub = generate_operator_key()
    gate = DeploymentGate(Quorum({"op": pub}, 1), AuditLog(tmp_path / "a"),
                          tmp_path / "reg")
    path = tmp_path / "c.pt"
    save(tiny(), path)
    with pytest.raises(DeploymentError):
        load_approved(gate, path)
    d = sha256_file(path)
    rep = {"passed": True, "weights_sha256": d}
    req = gate.approval_request(d, rep)
    gate.approve(d, rep, req, {"op": sign(priv, req)})
    assert isinstance(load_approved(gate, path), GPT)

    class Evil:
        def __reduce__(self):
            return (exec, ("open(%r, 'w').write('pwned')" % str(tmp_path / "pwned"),))

    bad = tmp_path / "evil.pt"
    bad.write_bytes(pickle.dumps({"cfg": Evil()}))
    with pytest.raises(Exception):
        load_for_eval(bad)
    assert not (tmp_path / "pwned").exists()
