# Opaque Tutorials

This directory contains Jupyter notebook tutorials for learning Opaque and differential privacy.

## Tutorial Overview

### [Tutorial: Gradient Clipping from Basics](01_gradient_clipping_from_basics.ipynb)

**Level**: Beginner to Intermediate
**Duration**: 60-75 minutes
**Prerequisites**: Basic PyTorch (tensors, `torch.nn`, optimizers)

A comprehensive tutorial that builds from first principles to Opaque's `clipped_grad()`:

**Learning progression**:
1. Train an MLP with **standard PyTorch** (averaged gradients)
2. Understand **why per-sample gradients matter** for gradient clipping
3. Compute per-sample gradients **manually** (naive iterative approach)
4. Use **functional transformations** (`grad` + `vmap`) for efficiency
5. Convert models to **functional form** with `make_functional()`
6. Use Opaque's **`clipped_grad()`** as a convenient wrapper

**Key Takeaway**: Graduate from manual per-sample gradient computation to using Opaque's clean, efficient `clipped_grad()` API - understanding every step along the way.

## Additional Resources

### Documentation
- [Opaque API Reference](../api/core/clipping.md)
- [Design Decisions](../development/design-decisions.md)
- [JAX-Privacy Comparison](../development/jax-privacy-comparison.md)

### Papers
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) (DP-SGD)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

### External Tutorials
- [PyTorch torch.func Guide](https://pytorch.org/tutorials/intermediate/functorch_tutorial.html)
- [DP-SGD Explained (Opacus blog)](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3)

---

## Contributing

Found an issue or have suggestions for improving the tutorials?

1. Open an issue on [GitHub](https://github.com/yourusername/opaque/issues)
2. Submit a PR with improvements
3. Share your own examples!

**Planned tutorials** (coming in future stages):
- [ ] Tutorial 3: Noise Injection Basics (Stage 2)
- [ ] Tutorial 4: Training with DP-SGD (Stage 2)
- [ ] Tutorial 5: Privacy Accounting (Stage 3)
- [ ] Tutorial 6: Complete Differential Privacy Workflow (Stage 2+)
- [ ] Tutorial 7: LoRA Fine-Tuning with DP (Stage 4)

---

## FAQ

**Q: Do I need a GPU to run these tutorials?**
A: No! All tutorials run fine on CPU. They use small datasets for educational purposes.

**Q: Can I use these notebooks in Google Colab?**
A: Yes! Upload the notebook and install Opaque:
```python
!pip install git+https://github.com/yourusername/opaque.git
```

**Q: Can I train differentially private models with this tutorial?**
A: Not yet! The tutorial covers gradient clipping, but DP requires noise injection (Stage 2). For now, you can use `clipped_grad()` for gradient norm regularization.

**Q: What's available now vs "coming soon"?**
A: As of Stage 1 (complete), `clipped_grad()` is fully functional for gradient clipping. Noise injection and privacy accounting are coming in Stages 2-3.

**Q: I found a bug in a tutorial. How do I report it?**
A: Please open an issue with:
- Notebook name
- Cell number
- Error message
- Your environment (Python version, OS)

---

**Happy Learning!** 🎓

Questions? Open an issue or discussion on GitHub.
