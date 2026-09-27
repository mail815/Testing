"""Checkpoint save/load that never executes code from the file.

``torch.load`` without ``weights_only=True`` unpickles arbitrary objects, so a
malicious checkpoint can run code on load. We store only tensors and plain
config values, and load only through the deployment gate's verified bytes.
"""

from __future__ import annotations

import io
import os

import torch

from guardian import DeploymentGate

from .gpt import GPT, GPTConfig


def save(model: GPT, path: str | os.PathLike[str]) -> None:
    tmp = f"{os.fspath(path)}.tmp"
    torch.save({"cfg": dict(model.cfg.__dict__), "model": model.state_dict()}, tmp)
    os.replace(tmp, path)  # atomic: never leave a half-written checkpoint


def _from_bytes(data: bytes) -> GPT:
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    m = GPT(GPTConfig(**state["cfg"]))
    m.load_state_dict(state["model"], strict=True)
    return m.eval()


def load_approved(gate: DeploymentGate, path: str | os.PathLike[str]) -> GPT:
    """Load weights only if their exact bytes are quorum-approved."""
    return _from_bytes(gate.require(path))


def load_for_eval(path: str | os.PathLike[str]) -> GPT:
    """Load (safely) for evaluation *before* approval. Never serve from this."""
    with open(path, "rb") as f:
        return _from_bytes(f.read())
