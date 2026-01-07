DEFAULT_STUDY_NAME="epiforecaster_hpo_v1"
STUDY_NAME="${1:-$DEFAULT_STUDY_NAME}"

REMOTE_ROOT="/home/bsc/bsc008913/EpiForecaster"
REMOTE_OPTUNA="$REMOTE_ROOT/outputs/optuna"
LOCAL_OPTUNA="./outputs/optuna"

mkdir -p "$LOCAL_OPTUNA"

rsync -avz --progress \
    dt:${REMOTE_OPTUNA}/${STUDY_NAME}.journal \
    "$LOCAL_OPTUNA/"

rsync -avz --progress \
    dt:${REMOTE_OPTUNA}/${STUDY_NAME} \
    "$LOCAL_OPTUNA/"
