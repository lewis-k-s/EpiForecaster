#!/bin/bash
rsync -avz --progress \
  --exclude "mobility.zarr/" \
  --exclude "raw_synthetic_observations.zarr/" \
  data/files/ dt:/home/bsc/bsc008913/EpiForecaster/data/files/
