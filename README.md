# MonoRCNN
MonoRCNN is a monocular 3D object detection method for automonous driving, published at ICCV 2021. This project is an implementation of MonoRCNN.

<img src='image/KITTI-testset.png' width=809 height=188>

## Link
* [ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/html/Shi_Geometry-Based_Distance_Decomposition_for_Monocular_3D_Object_Detection_ICCV_2021_paper.html)
* [KITTI video demo](https://www.youtube.com/watch?v=46lToJSagcg)

## Installation

* Python 3.6
* PyTorch 1.5.0 
* Detectron2 0.1.3 (included in this project, see [INSTALL.md](INSTALL.md) to install it.)

## Dataset Preparation
* [KITTI](projects/KITTI/README.md)

## Test
```
cd projects/MonoRCNN
./main.py --config-file config/MonoRCNN_KITTI.yaml --num-gpus 1 --resume --eval-only
```
Set `TEST.VISUALIZE` in [MonoRCNN_KITTI.yaml](projects/MonoRCNN/config/MonoRCNN_KITTI.yaml) as `True` to visualize 3D object detection results.

## Training
```
cd projects/MonoRCNN
./main.py --config-file config/MonoRCNN_KITTI.yaml --num-gpus 1
```

## Citation
If you find this project useful in your research, please consider citing:

```
@inproceedings{MonoRCNN_ICCV21,
    title = {Geometry-based Distance Decomposition for Monocular 3D Object Detection},
    author = {Xuepeng Shi and Qi Ye and 
              Xiaozhi Chen and Chuangrong Chen and 
              Zhixiang Chen and Tae-Kyun Kim},
    booktitle = {ICCV},
    year = {2021},
}
```

## Contact
x.shi19@imperial.ac.uk

## Acknowledgement
I build this project based on [Detectron2](https://github.com/facebookresearch/detectron2), and also refer to [M3D-RPN](https://github.com/garrickbrazil/M3D-RPN) and [MMDetection](https://github.com/open-mmlab/mmdetection).
