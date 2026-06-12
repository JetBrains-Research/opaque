FROM ubuntu:24.04

# Run as the non-root `dev` user that the common-utils feature creates. Its uid is
# pinned to 1001 in devcontainer.json (userUid/userGid) — keep these in sync.
# The devcontainer build restores this base USER as the final image USER, making
# the published image non-root by default: required by k8s/Kyverno
# (require-run-as-nonroot rejects root/non-numeric users) and it keeps cargo/rustup
# (owned by dev) writable. The uid need not exist yet here — the feature creates it.
#
# Use the bare uid (not "1001:1001"): because `dev` has a passwd entry, this resolves
# to gid 1001 AND loads dev's supplementary groups (rustlang/python/pipx/docker).
# Specifying "uid:gid" makes Docker skip NSS group setup and drop those groups.
USER 1001
