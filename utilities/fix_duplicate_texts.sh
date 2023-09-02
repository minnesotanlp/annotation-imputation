# Check if python3 is available
if command -v python3 &>/dev/null; then
    python_command="python3"
# Check if python is available
elif command -v python &>/dev/null; then
    python_command="python"
else
    echo "Error: Neither python nor python3 is installed on this system."
    exit 1
fi

DATASETS=("politeness_binary" "SChem" "SChem5Labels" "SBIC" "Sentiment" "ghc" "politeness")
DATASET_LOCATION="../datasets"

for dataset in DATASETS
do
    $python_command fix_duplicate_texts.py --dataset_name ${dataset} --dataset_location ${DATASET_LOCATION}