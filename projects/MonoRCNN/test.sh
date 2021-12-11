export CUDA_VISIBLE_DEVICES=0
./main.py --config-file config/MonoRCNN_KITTI.yaml --num-gpus 1 --resume --eval-only