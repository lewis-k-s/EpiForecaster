#!/bin/bash
rsync -avz --progress . dt:/home/bsc/bsc008913/EpiForecaster \
  --exclude-from=".gitignore" \
  --exclude ".git" \
  --exclude "data/processed" \
  --filter="+ /outputs/" \
  --filter="+ /outputs/region_embeddings/" \
  --filter="+ /outputs/region_embeddings/***" \
  --filter="- /outputs/*/" \
  --filter="- /outputs/***"
