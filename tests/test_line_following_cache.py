import pytest
import torch

from coursekit.labs.line_following_cache import ahead


def quadratic(x):
    return x.square() + 1


def test_saved_kernel_uses_current_inputs_and_recovers_from_damage(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSEKIT_COMPILE_CACHE", str(tmp_path))

    def kernel():
        return torch.compile(quadratic, fullgraph=True, backend="eager")

    first = ahead(kernel(), 2)
    torch.testing.assert_close(first(torch.tensor([2.0, 3.0])), torch.tensor([5.0, 10.0]))
    saved = kernel()
    monkeypatch.setattr(
        saved,
        "aot_compile",
        lambda *args: pytest.fail("A valid saved kernel must not compile again"),
    )
    loaded = ahead(saved, 2)
    torch.testing.assert_close(loaded(torch.tensor([4.0, 5.0])), torch.tensor([17.0, 26.0]))
    [artifact] = tmp_path.rglob("*.pt")
    artifact.write_bytes(b"interrupted or damaged artifact")
    with pytest.warns(UserWarning, match="Rebuilding robot kernel cache"):
        restored = ahead(kernel(), 2)(torch.tensor([6.0, 7.0]))
    torch.testing.assert_close(restored, torch.tensor([37.0, 50.0]))
    torch.testing.assert_close(
        ahead(saved, 2)(torch.tensor([2.0, 3.0])), torch.tensor([5.0, 10.0])
    )
    assert not list(tmp_path.rglob("*.tmp"))
