#!/usr/bin/env zsh
set -euo pipefail

# Default values
TRAIN_CONFIG="${1:-configs/train_epifor_full.yaml}"
PREPROCESS_CONFIG="${2:-configs/preprocess_full.yaml}"
OUTPUT_DIR="${3:-outputs/dataviz}"

mkdir -p "$OUTPUT_DIR"

echo "EpiForecaster Data Visualization"
echo "Output: $OUTPUT_DIR"

declare -a analyses=(
    "dataviz/cases_sparsity_analysis.py --train-config $TRAIN_CONFIG --preprocess-config $PREPROCESS_CONFIG --output-dir $OUTPUT_DIR/cases_sparsity"
    "dataviz/edge_weight_analysis.py --output-dir $OUTPUT_DIR/edge_weight"
    "dataviz/khop_neighbors.py --config $TRAIN_CONFIG --output-dir $OUTPUT_DIR/khop_neighbors --k 3"
    "dataviz/mobility_lockdown_analysis.py --config $TRAIN_CONFIG --output-dir $OUTPUT_DIR/mobility_lockdown"
    "dataviz/mobility_regime_analysis.py --config $TRAIN_CONFIG --output-dir $OUTPUT_DIR/mobility_regime"
    "dataviz/neighborhood_global_regression.py --config $TRAIN_CONFIG --output-dir $OUTPUT_DIR/neighborhood_regression --target cases"
    "dataviz/neighborhood_trace_density.py --config $TRAIN_CONFIG --output-dir $OUTPUT_DIR/neighborhood_density --split all"
    "dataviz/sparsity_analysis.py --config $TRAIN_CONFIG --geo-path data/files/geo/fl_municipios_catalonia.geojson --output-dir $OUTPUT_DIR/sparsity --window-size 7"
    "dataviz/te_sanity_check.py --dataset data/processed/full_v2.zarr --target cases --source mobility --lags 7"
    "dataviz/timeseries_analysis.py --config $TRAIN_CONFIG --output-dir $OUTPUT_DIR/timeseries"
    "dataviz/timeseries_filtered.py --config $TRAIN_CONFIG --output-dir $OUTPUT_DIR/timeseries_filtered"
)

for script in "${analyses[@]}"; do
    echo "Running: $script"
    uv run python $script || echo "Failed: $script"
done

echo "Done. Outputs in $OUTPUT_DIR"
