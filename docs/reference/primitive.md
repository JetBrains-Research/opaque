# opaque.primitive

The authoring surface for backend providers: declare dispatched primitives,
register implementations, and validate a provider against the portable core
profile. Library users never need this page; provider authors and
integrators do. See the contributor guide in
[`docs/development/backend-providers.md`](../development/backend-providers.md)
for the end-to-end authoring walkthrough.

## Authoring

::: opaque.primitive.primitive
    options:
        show_source: true
        heading_level: 3

::: opaque.primitive.Primitive
    options:
        show_source: false
        heading_level: 3

::: opaque.primitive.BackendProvider
    options:
        show_source: false
        heading_level: 3

## Core profile

::: opaque.primitive.CORE_PROFILE_VERSION
    options:
        heading_level: 3

::: opaque.primitive.PrimitiveTier
    options:
        heading_level: 3
