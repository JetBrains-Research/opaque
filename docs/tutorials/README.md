# Opaque Tutorials

This directory contains Jupyter notebook tutorials for learning Opaque and differential privacy.

## Tutorial Overview

### [Tutorial 01: Gradient Clipping from Basics](01_gradient_clipping_from_basics.ipynb)

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

---

### [Tutorial 02: Differential Privacy - Noise and Accounting](02_differential_privacy_noise_and_accounting.ipynb) ✨ NEW!

**Level**: Intermediate
**Duration**: 60-75 minutes
**Prerequisites**: Tutorial 01, Basic understanding of privacy concepts

Builds on Tutorial 01 to implement complete DP-SGD with noise injection and privacy accounting:

**Learning objectives**:

1. Understand why **noise is needed** for differential privacy
2. Use `add_gaussian_noise()` to add **calibrated Gaussian noise**
3. Track privacy with **accountants** (`PLDAccountant`, `RDPAccountant`)
4. **Calibrate noise multipliers** for target privacy budgets
5. Implement **complete DP-SGD training loop**
6. Explore the **privacy-utility tradeoff**

**Key Takeaway**: Complete DP-SGD workflow from calibration to training with formal privacy guarantees (ε, δ).

---

### [Tutorial 03: Complete DP-SGD Training Loop](03_complete_dp_sgd_training.ipynb)

**Level**: Intermediate
**Duration**: 45-60 minutes
**Prerequisites**: Tutorial 01 (Gradient Clipping), Tutorial 02 (Noise & Accounting)

Puts everything together to implement a complete DP-SGD training loop:

**What you'll build**:

1. **See the minimal difference**: Non-DP vs DP-SGD (just 2 lines!)
2. Complete **DP-SGD training loop**
3. Compare **DP-SGD vs non-private training**
4. Visualize **privacy-utility tradeoffs**
5. Track **privacy budget** throughout training

**Key Takeaway**: End-to-end DP-SGD implementation with clear side-by-side comparison showing DP is just a 2-line
change!

---

### [Tutorial 04: DP Optimizers - From Manual to Production-Ready](04_dp_optimizers.ipynb) ✨ NEW!

**Level**: Intermediate to Advanced
**Duration**: 60-75 minutes
**Prerequisites**: Tutorials 01-03 (Gradient Clipping, Noise, Manual DP-SGD)

Learn production-ready DP optimizers that handle noise, updates, and accounting automatically:

**What you'll learn**:

1. **Recap**: Manual DP-SGD from Tutorial 03
2. **DP-SGD optimizer**: Production-ready with automatic noise + accounting
3. **DP-AdamW**: Adaptive learning rates + weight decay for better convergence
4. **DP-Adam-AC** (Extra): Adaptive clipping for optimal privacy-utility tradeoff

**Key Takeaway**: Opaque's optimizers make DP training as easy as standard PyTorch training - just one line per step
instead of manual noise + updates + accounting!

---

### [Tutorial 05: DP-SGD for LoRA Fine-Tuning with HuggingFace](05_lora_huggingface_dp_training.ipynb) ✨ NEW!

**Level**: Advanced
**Duration**: 60-90 minutes
**Prerequisites**: Tutorials 01-04, Basic familiarity with transformers and LoRA

Apply DP-SGD to real-world use case: fine-tuning large language models with LoRA:

**What you'll learn**:

1. Why **LoRA is ideal for DP** fine-tuning (smaller gradients, better SNR)
2. Apply **LoRA to HuggingFace models** with PEFT
3. Convert models to **functional form** with frozen + trainable parameters
4. Use **DP-AdamW optimizer** from Tutorial 04 for production training
5. Use **PyTorch DataLoader** for proper batch handling
6. Train with **fixed batch sampling** (RDP accountant)
7. **Practical hyperparameter guidance** for stable DP-SGD training

**Key Takeaway**: Production-ready DP-SGD for HuggingFace models with LoRA using Opaque's DP-AdamW optimizer.

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

**Completed tutorials**:

- [x] Tutorial 01: Gradient Clipping from Basics (Stage 1)
- [x] Tutorial 02: Differential Privacy - Noise and Accounting (Stage 2)
- [x] Tutorial 03: Complete DP-SGD Training Loop (Stage 3)
- [x] Tutorial 04: DP Optimizers - From Manual to Production-Ready (Stage 3)
- [x] Tutorial 05: DP-SGD for LoRA Fine-Tuning with HuggingFace (Stage 3)

**Planned tutorials** (coming in future stages):

- [ ] Tutorial 06: Advanced Privacy Techniques (Poisson sampling, microbatching, etc.) (Stage 4+)
- [ ] Tutorial 07: Multi-GPU DP Training (Stage 5+)

---

## FAQ

**Q: Do I need a GPU to run these tutorials?**
A: No! All tutorials run fine on CPU. They use small datasets for educational purposes.

**Q: Can I use these notebooks in Google Colab?**
A: Yes! Upload the notebook and install Opaque:
```python
!pip install git+https://github.com/yourusername/opaque.git
```

**Q: Can I train differentially private models with these tutorials?**
A: Yes! Tutorial 02 covers complete DP-SGD with noise injection and privacy accounting. You can train models with
formal (ε, δ)-DP guarantees.

**Q: What's available now vs "coming soon"?**
A: As of Stage 2 (complete), all core DP-SGD components are ready:

- ✅ Gradient clipping (`clipped_grad()`)
- ✅ Noise injection (`add_gaussian_noise()`)
- ✅ Privacy accounting (`PLDAccountant`, `RDPAccountant`)
- ✅ Calibration functions for target privacy
- 🔜 High-level API and LoRA integration (Stages 3-4)

**Q: I found a bug in a tutorial. How do I report it?**
A: Please open an issue with:
- Notebook name
- Cell number
- Error message
- Your environment (Python version, OS)

---

**Happy Learning!** 🎓

Questions? Open an issue or discussion on GitHub.
