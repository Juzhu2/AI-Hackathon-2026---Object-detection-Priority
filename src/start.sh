#!/usr/bin/env bash
# One command to start everything: ./start.sh   (extra options pass through, e.g. ./start.sh --camera 1)
cd "$(dirname "$0")"
source ~/venv/bin/activate
exec python run_all.py "$@"
