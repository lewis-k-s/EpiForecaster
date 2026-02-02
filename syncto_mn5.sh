#!/bin/bash
rsync -avz --progress . dt:/home/bsc/bsc008913/EpiForecaster \
  --filter="+ /outputs/" \
  --filter="+ /outputs/region_embeddings/" \
  --filter="+ /outputs/region_embeddings/***" \
  --filter="- /outputs/*/" \
  --filter="- /outputs/***" \
  --exclude "data/processed" \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".mypy_cache" \
  --exclude ".pytest_cache" \
  --exclude ".ruff_cache" \
  --exclude "EpiForecaster.egg-info"
