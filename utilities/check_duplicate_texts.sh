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

datasets=(ghc Sentiment SBIC SChem SChem5Labels)

for dataset in ${datasets[@]}
do
    echo "Checking duplicate texts in ${dataset}..."
    $python_command check_duplicate_texts.py --file datasets/${dataset}_texts.npy
done