DEFAULT_STUDY_NAME="epiforecaster_hpo_v1"
STUDY_NAME="${1:-$DEFAULT_STUDY_NAME}"

REMOTE_ROOT="/home/bsc/bsc008913/EpiForecaster"
REMOTE_HPSEARCH="$REMOTE_ROOT/outputs/hpsearch"
LOCAL_HPSEARCH="./outputs/hpsearch"

mkdir -p "$LOCAL_HPSEARCH"

rsync -avz --progress \
    dt:${REMOTE_HPSEARCH}/${STUDY_NAME}.journal \
    "$LOCAL_HPSEARCH/"

rsync -avz --progress \
    dt:${REMOTE_HPSEARCH}/${STUDY_NAME} \
    "$LOCAL_HPSEARCH/"
