## KITTI
We take the KITTI val1 split as an example to show how to prepare the training and test subset.

1. Download the training dataset of the KITTI monocular 3D object detection task, and organize the downloaded files as follows:
```
├── projects
│   ├── KITTI
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
```

2. Prepare the training and test subset of the KITTI val1 split:
```
python prepare_KITTI.py
```

3. Compile the evaluation code: 
```
cd val1/cpp
sh build.sh
```