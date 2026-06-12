FROM ubuntu:24.04

# Run as the non-root `dev` user that the common-utils feature creates. Its uid is
# pinned to 1001 in devcontainer.json (userUid/userGid) — keep these in sync.
# The devcontainer build restores this base USER as the final image USER, making
# the published image non-root by default: required by k8s/Kyverno
# (require-run-as-nonroot rejects root/non-numeric users) and it keeps cargo/rustup
# (owned by dev) writable. The uid need not exist yet here — the feature creates it.
USER 1001
