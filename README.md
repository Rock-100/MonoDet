# MonoRCNN
MonoRCNN is a monocular 3D object detection method for autonomous driving, published at ICCV 2021 and WACV 2023. This project is an implementation of MonoRCNN.

## Related Link
* [Video demo (KITTI)](https://www.youtube.com/watch?v=46lToJSagcg)
* [Paper (ICCV 2021)](https://arxiv.org/abs/2104.03775)
* [Paper (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/html/Shi_Multivariate_Probabilistic_Monocular_3D_Object_Detection_WACV_2023_paper.html)
* [KITTI benchmark (ICCV 2021)](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=06e0aacbeb8de57a09d2a3557907064aa501048f)
* [KITTI benchmark (WACV 2023)](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=5bbb320aee42eee314697370b8183217d61e8141)

## Visualization
* KITTI
<img src='images/KITTI_testset_000095.png' width=805>
<img src='images/KITTI_testset_000181.png' width=805>

* WAYMO
<p float="left">
<img src='images/WAYMO_valset_057159.png' width=400>
<img src='images/WAYMO_valset_060794.png' width=400>
</p>

## Installation

* Python 3.6
* PyTorch 1.5.0 
* Detectron2 0.1.3 

Please use the Detectron2 included in this project. To ignore fully occluded objects during training, [`build.py`](detectron2/data/build.py#L55), [`rpn.py`](detectron2/modeling/proposal_generator/rpn.py#L292), and [`roi_heads.py`](detectron2/modeling/roi_heads/roi_heads.py#L272) have been modified.

## Dataset Preparation
* [KITTI](projects/KITTI/README.md)

## Model & Log
* [KITTI val split](https://drive.google.com/file/d/1OEHSLqg7goXb483GRLvM-z96ZDzN7wdI/view?usp=sharing)

Organize the downloaded files as follows:
```
├── projects
│   ├── MonoRCNN
│   │   ├── output
│   │   │   ├── model
│   │   │   ├── log.txt
│   │   │   ├── ...
```

* [Waymo](https://drive.google.com/file/d/1fL3E3yqZbRd3va3kN-1j-1YlX61-Qo13/view?usp=sharing)

## Test
```
cd projects/MonoRCNN
./main.py --config-file config/MonoRCNN_KITTI.yaml --num-gpus 1 --resume --eval-only
```
Set [`VISUALIZE`](projects/MonoRCNN/config/MonoRCNN_KITTI.yaml#L13) as `True` to visualize 3D object detection results (saved in `output/evaluation/test/visualization`).

## Training
```
cd projects/MonoRCNN
./main.py --config-file config/MonoRCNN_KITTI.yaml --num-gpus 1
```

## Citation
If you find this project useful in your research, please cite:

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
```
@inproceedings{MonoRCNN_WACV23,
    title = {Multivariate Probabilistic Monocular 3D Object Detection},
    author = {Xuepeng Shi and Zhixiang Chen and Tae-Kyun Kim},
    booktitle = {WACV},
    year = {2023},
}
```

## Contact
x.shi19@imperial.ac.uk

## Acknowledgement
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [M3D-RPN](https://github.com/garrickbrazil/M3D-RPN)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
