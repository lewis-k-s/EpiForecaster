DEFAULT_EXPERIMENT_NAME="mn5_epiforecaster_full"
EXPERIMENT_NAME="${1:-$DEFAULT_EXPERIMENT_NAME}"

rsync -avz --progress \
    dt:/home/bsc/bsc008913/EpiForecaster/outputs/training/$EXPERIMENT_NAME \
    ./outputs/training/