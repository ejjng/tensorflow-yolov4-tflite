search_dir=./checkpoints/models/
for dir in "$search_dir"/*/
do
    CUDA_VISIBLE_DEVICES="" python convert_tflite.py --name "$(basename -- $dir)" &
done
wait