"""Tests for make_functional with partition_trainable feature."""

import pytest
import torch
import torch.nn as nn

from opaque.utils import make_functional, merge


class TestMakeFunctionalPartition:
    """Tests for partition_trainable parameter."""

    def test_partition_simple_model(self):
        """Test partitioning a simple model."""
        model = nn.Linear(10, 5)

        # Freeze weight, keep bias trainable
        model.weight.requires_grad = False
        model.bias.requires_grad = True

        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

        assert "bias" in trainable
        assert "weight" in frozen
        assert "bias" not in frozen
        assert "weight" not in trainable

    def test_partition_all_trainable(self):
        """Test when all parameters are trainable."""
        model = nn.Linear(10, 5)

        # All trainable (default)
        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

        assert "weight" in trainable
        assert "bias" in trainable
        assert len(frozen) == 0

    def test_partition_all_frozen(self):
        """Test when all parameters are frozen."""
        model = nn.Linear(10, 5)

        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

        assert "weight" in frozen
        assert "bias" in frozen
        assert len(trainable) == 0

    def test_partition_sequential_model(self):
        """Test partitioning a sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 5, bias=False),
            nn.Linear(5, 2, bias=False),
        )

        # Freeze first layer, keep second trainable
        model[0].weight.requires_grad = False
        model[1].weight.requires_grad = True

        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

        assert "1.weight" in trainable  # Second layer
        assert "0.weight" in frozen  # First layer
        assert len(trainable) == 1
        assert len(frozen) == 1

    def test_fmodel_with_dict_params(self):
        """Test that fmodel works with dict parameters."""
        model = nn.Linear(10, 5)
        model.weight.requires_grad = False
        model.bias.requires_grad = True

        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

        # Merge and use
        all_params = merge(frozen, trainable)
        x = torch.randn(3, 10)
        output = fmodel(all_params, x)

        assert output.shape == (3, 5)

    def test_forward_pass_with_partitioned_params(self):
        """Test forward pass produces correct results."""
        model = nn.Linear(10, 5)

        # Save original forward pass result
        x = torch.randn(3, 10)
        with torch.no_grad():
            expected = model(x)

        # Partition
        model.weight.requires_grad = False
        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

        # Forward with merged params
        all_params = merge(frozen, trainable)
        output = fmodel(all_params, x)

        # Should match original
        assert torch.allclose(output, expected, atol=1e-5)

    def test_update_only_trainable(self):
        """Test that we can update only trainable params."""
        model = nn.Linear(10, 5)
        model.weight.requires_grad = False
        model.bias.requires_grad = True

        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

        # Save original values
        original_trainable_bias = trainable["bias"].clone()
        original_frozen_weight = frozen["weight"].clone()

        # Update trainable
        trainable["bias"] = trainable["bias"] + 1.0

        # Frozen should be unchanged
        assert torch.allclose(frozen["weight"], original_frozen_weight)

        # Trainable should be changed
        assert not torch.allclose(trainable["bias"], original_trainable_bias)

    def test_partition_with_nested_modules(self):
        """Test partitioning with nested module structure."""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 5)
                self.decoder = nn.Linear(5, 2)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = NestedModel()

        # Freeze encoder
        model.encoder.weight.requires_grad = False
        model.encoder.bias.requires_grad = False

        # Keep decoder trainable
        model.decoder.weight.requires_grad = True
        model.decoder.bias.requires_grad = True

        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

        assert "decoder.weight" in trainable
        assert "decoder.bias" in trainable
        assert "encoder.weight" in frozen
        assert "encoder.bias" in frozen

    def test_disable_autograd_with_partition(self):
        """Test that disable_autograd_tracking works with partitioning."""
        model = nn.Linear(10, 5)
        model.weight.requires_grad = False

        fmodel, trainable, frozen = make_functional(
            model, partition_trainable=True, disable_autograd_tracking=True
        )

        # Parameters should be detached
        assert not trainable["bias"].requires_grad
        assert not frozen["weight"].requires_grad

    def test_backward_compatibility(self):
        """Test that partition_trainable=False maintains original behavior."""
        model = nn.Linear(10, 5)

        # Original behavior
        fmodel, params = make_functional(model, partition_trainable=False)

        assert isinstance(params, tuple)
        assert len(params) == 2  # weight and bias

        # Should work
        x = torch.randn(3, 10)
        output = fmodel(params, x)
        assert output.shape == (3, 5)


class TestLoRAWorkflowWithMakeFunctional:
    """Integration tests for LoRA workflow using make_functional."""

    def test_typical_lora_workflow(self):
        """Test typical LoRA workflow with make_functional."""

        # 1. Create model with some layers frozen
        model = nn.Sequential(
            nn.Linear(20, 10, bias=False),  # Pretrained backbone
            nn.Linear(10, 5, bias=False),  # LoRA adapter
            nn.Linear(5, 2, bias=False),  # LoRA adapter
        )

        # Freeze backbone
        model[0].weight.requires_grad = False

        # Keep adapters trainable
        model[1].weight.requires_grad = True
        model[2].weight.requires_grad = True

        # 2. Convert to functional with partitioning
        fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

        # 3. Verify partition
        assert len(trainable) == 2  # Two adapter layers
        assert len(frozen) == 1  # One frozen layer
        assert "0.weight" in frozen
        assert "1.weight" in trainable
        assert "2.weight" in trainable

        # 4. Simulate training update (only trainable params)
        for key in trainable:
            trainable[key] = trainable[key] - 0.01 * torch.randn_like(trainable[key])

        # 5. Merge for forward pass
        all_params = merge(frozen, trainable)

        # 6. Forward pass works
        x = torch.randn(3, 20)
        output = fmodel(all_params, x)

        assert output.shape == (3, 2)

    def test_gradient_computation_on_trainable_only(self):
        """Test that gradients are only computed for trainable params."""
        from torch.func import grad

        model = nn.Linear(10, 5)
        model.weight.requires_grad = False
        model.bias.requires_grad = True

        fmodel, trainable, frozen = make_functional(
            model, partition_trainable=True, disable_autograd_tracking=True
        )

        def loss_fn(train_params, x, y):
            all_params = merge(frozen, train_params)
            pred = fmodel(all_params, x)
            return ((pred - y) ** 2).mean()

        x = torch.randn(3, 10)
        y = torch.randn(3, 5)

        # Compute gradients only w.r.t. trainable
        grads = grad(loss_fn)(trainable, x, y)

        # Should only have gradient for bias
        assert "bias" in grads
        assert "weight" not in grads
        assert grads["bias"].shape == trainable["bias"].shape

    def test_full_dp_training_loop(self):
        """Test full DP training loop with partitioned params."""
        from opaque import clipped_grad, dp_sgd

        # Create model
        model = nn.Linear(10, 1)
        model.weight.requires_grad = False  # Freeze weight
        model.bias.requires_grad = True  # Train only bias

        # Convert with partitioning
        fmodel, trainable, frozen = make_functional(
            model, partition_trainable=True, disable_autograd_tracking=True
        )

        # Create DP optimizer (only for trainable!)
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
            l2_clip_norm=1.0,
            noise_multiplier=0.1,  # Low noise for test
            sample_rate=0.1,
            target_delta=1e-5,
        )

        state = init_fn(trainable)

        # Loss function
        def loss_fn(train_params, x, y):
            all_params = merge(frozen, train_params)
            pred = fmodel(all_params, x)
            return ((pred - y) ** 2).mean()

        # Clipped gradient function
        clipped_grad_fn = clipped_grad(loss_fn, argnums=0, batch_argnums=(1, 2), l2_clip_norm=1.0)

        # Generate data
        X = torch.randn(20, 10)  # noqa: N806
        y = torch.randn(20, 1)

        # Training step
        grads = clipped_grad_fn(trainable, X, y)
        trainable, state, metrics = step_fn(trainable, grads, state)

        # Verify
        assert state.step == 1
        assert metrics["epsilon"] > 0
        assert "bias" in grads
        assert "weight" not in grads  # Not trainable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
