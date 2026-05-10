#!/bin/bash
# Backward-compatible wrapper: delegates to the config-driven dispatch CLI.
exec "$(dirname "$0")/cluster_dispatch.sh" "$@"
