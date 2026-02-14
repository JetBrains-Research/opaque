# Privacy Accounting

The `opaque.accounting` module provides a functional API for tracking and querying privacy budgets during DP training.

## Overview

Privacy accounting tracks how privacy degrades across training steps. Opaque uses:

- **Immutable state**: Privacy state is never modified, only new states created
- **Functional composition**: Pure functions compose privacy guarantees
- **Multiple metrics**: Query (ε, δ)-DP, f-DP advantage, or (α, β) error rates
- **Tight bounds**: Privacy Loss Distribution (PLD) accounting for optimal bounds

**Key functions**:

- `create()` - Initialize privacy state
- `compose_*()` - Compose privacy over training steps
- `get_*()` - Query privacy guarantees
- `find_noise_multiplier_for_*()` - Calibrate noise for target privacy

**See also**: [Privacy Accounting User Guide](../user-guide/accounting.md)

<!-- TODO: Uncomment when accounting module is implemented
## Composition Functions

::: opaque.accounting.composition

## Privacy Queries

::: opaque.accounting.queries

## Calibration Functions

::: opaque.accounting.calibration
-->
