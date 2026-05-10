#!/bin/bash
REMOTE_ROOT="dt:/home/bsc/bsc008913/EpiForecaster"
rsync -avz --progress \
  --exclude "mobility.zarr/" \
  --exclude "raw_synthetic_observations.zarr/" \
  "${REMOTE_ROOT}/data/files/" ./data/files/
