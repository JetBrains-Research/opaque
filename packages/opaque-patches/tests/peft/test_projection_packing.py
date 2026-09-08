import torch

from opaque.api.patches.peft.components._packing import (
    _projection_packs,
    _register_projection_pack,
)


class _PackOwner(torch.nn.Module):
    def __init__(self, *, register_pack=True):
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(5, 4), requires_grad=False),
                torch.nn.Parameter(torch.randn(2, 4), requires_grad=False),
                torch.nn.Parameter(torch.randn(3, 4), requires_grad=False),
            ]
        )
        self.biases = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(5), requires_grad=False),
                torch.nn.Parameter(torch.randn(2), requires_grad=False),
                torch.nn.Parameter(torch.randn(3), requires_grad=False),
            ]
        )
        self.adapter_a = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(2, 4)) for _ in range(3)]
        )
        self.adapter_b = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(5, 2)),
                torch.nn.Parameter(torch.randn(2, 2)),
                torch.nn.Parameter(torch.randn(3, 2)),
            ]
        )
        if register_pack:
            _register_projection_pack(
                self, "qkv", ("weight", "bias", "adapter_a", "adapter_b")
            )

    def sources(self):
        adapters = tuple(
            (A.T, B.T, scaling)
            for A, B, scaling in zip(
                self.adapter_a, self.adapter_b, (0.5, 1.0, 2.0), strict=True
            )
        )
        return tuple(self.weights), tuple(self.biases), adapters

    def forward(self, scale=None):
        packed = _projection_packs(self, "qkv", *self.sources())[0]
        return packed if scale is None else packed.sum() * scale


def test_projection_pack_cache_reuses_and_refreshes_nonpersistent_buffers():
    owner = _PackOwner()
    weights, biases, adapters = owner.sources()
    first = _projection_packs(owner, "qkv", weights, biases, adapters)
    second = _projection_packs(owner, "qkv", weights, biases, adapters)

    assert all(left is right for left, right in zip(first, second, strict=True))
    assert not any("_opaque_qkv_" in key for key in owner.state_dict())
    torch.testing.assert_close(first[0], torch.cat(weights))
    torch.testing.assert_close(first[1], torch.cat(biases))

    expected_A = torch.cat([A for A, _, _ in adapters], dim=1)
    expected_B = torch.block_diag(*(B * scaling for _, B, scaling in adapters))
    torch.testing.assert_close(first[2], expected_A)
    torch.testing.assert_close(first[3], expected_B)

    old_adapter_pack = first[3]
    with torch.no_grad():
        owner.adapter_b[1].add_(1)
    refreshed = _projection_packs(owner, "qkv", *owner.sources())
    assert refreshed[3] is not old_adapter_pack
    torch.testing.assert_close(
        refreshed[3],
        torch.block_diag(*(B * scaling for _, B, scaling in owner.sources()[2])),
    )

    state = owner.state_dict()
    state["weights.0"] = torch.full_like(state["weights.0"], 7)
    old_weight_pack = refreshed[0]
    owner.load_state_dict(state)
    loaded = _projection_packs(owner, "qkv", *owner.sources())
    assert loaded[0] is not old_weight_pack
    assert torch.all(loaded[0][:5] == 7)

    owner.to(dtype=torch.float64)
    converted = _projection_packs(owner, "qkv", *owner.sources())
    assert all(pack is None or pack.dtype == torch.float64 for pack in converted)


def test_projection_pack_falls_back_and_does_not_cache_external_tensors():
    owner = _PackOwner(register_pack=False)
    cached = _projection_packs(owner, "qkv", *owner.sources())
    assert owner._opaque_projection_pack_keys
    assert owner._opaque_qkv_weight_pack.numel() > 0
    cached_weight = cached[0]
    weights, biases, adapters = owner.sources()

    incompatible = list(adapters)
    incompatible[1] = (
        torch.randn(4, 3),
        torch.randn(3, 2),
        incompatible[1][2],
    )
    result = _projection_packs(owner, "qkv", weights, biases, tuple(incompatible))
    assert result[2:] == (None, None)

    external_weights = tuple(weight.detach().clone() for weight in weights)
    external = _projection_packs(owner, "qkv", external_weights, biases, adapters)
    assert external[0] is not cached_weight
    assert owner._opaque_qkv_weight_pack is cached_weight

    replacements = {
        name: parameter.detach().clone() for name, parameter in owner.named_parameters()
    }
    replacements["weights.0"] = torch.full_like(replacements["weights.0"], 11)
    functional_pack = torch.func.functional_call(owner, replacements, ())
    assert torch.all(functional_pack[:5] == 11)
    assert owner._opaque_qkv_weight_pack is cached_weight

    def loss(parameters, scale):
        return torch.func.functional_call(owner, parameters, (scale,))

    torch.vmap(torch.func.grad(loss), in_dims=(None, 0))(replacements, torch.ones(2))
    assert owner._opaque_qkv_weight_pack is cached_weight
