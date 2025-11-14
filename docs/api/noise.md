# Noise Injection

The `opaque.noise` module provides functions for adding calibrated Gaussian noise to gradients for differential privacy.

## Overview

After clipping gradients, DP-SGD requires adding Gaussian noise proportional to the clip norm and noise multiplier. The
noise obscures individual contributions, providing the actual privacy guarantee.

**Key function**: `add_gaussian_noise()` - Stateless noise addition with PyTree support

**See also**: [Noise Addition User Guide](../user-guide/noise.md)

## API Documentation

::: opaque.noise.gaussian
