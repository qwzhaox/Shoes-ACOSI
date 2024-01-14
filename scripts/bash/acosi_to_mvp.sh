task=$1

if [ $task = "acos" ]; then
    echo "acos"
elif [ $task = "acosi" ]; then
    echo "acosi"
else
    echo "task not found"
    exit 1
fi

if [ ! -d "../multi-view-prompting-acosi/data/$task/shoes" ]; then
    mkdir -p ../multi-view-prompting-acosi/data/$task/shoes
fi

cp data/mvp_dataset/train.txt ../multi-view-prompting-acosi/data/$task/shoes/train.txt
cp data/mvp_dataset/test.txt ../multi-view-prompting-acosi/data/$task/shoes/test.txt
cp data/mvp_dataset/dev.txt ../multi-view-prompting-acosi/data/$task/shoes/dev.txt