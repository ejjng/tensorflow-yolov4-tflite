model_dir=./checkpoints/tflite_models/
image_dir=./data/mixed_images/
for model_name in "$model_dir"/*
do
for image_name in "$image_dir"/*
do
    # echo "$(basename -- $model_name)"
    # echo "$(basename -- $image_name)"
    CUDA_VISIBLE_DEVICES="" python detect.py --name "$(basename -- $model_name)" \
        --image_name "$(basename -- $image_name)" &
done
done
wait
