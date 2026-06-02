#!/usr/bin/env bash
cd "$(dirname "$0")/.."
exec .venv/bin/pool-fool-calibrate doctor "$@"
