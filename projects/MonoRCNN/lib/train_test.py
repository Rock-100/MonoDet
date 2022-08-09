#!/usr/bin/env python
import os
import contextlib
import io
import logging
import copy
import torch
import random
import time
import datetime
import subprocess
import numpy as np
import re
import cv2
cv2.setNumThreads(0)

from contextlib import contextmanager
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager, file_lock

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import CfgNode as CN
from detectron2.structures import BoxMode, Boxes, Instances
from detectron2.utils.comm import get_world_size
from detectron2.data import (
    DatasetCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
    transforms as T,
    detection_utils as utils,
)

from lib.augmentation import (
    aug_resize,
    aug_flip,
    aug_pmd,
)

from lib.transformation import (
    compute_box_3d,
    project_to_image,
    alpha2rot,
    rot2alpha,
)

from lib.visualization import (
    draw_image,
    init_bev,
    draw_bev,
)

logger = logging.getLogger('detectron2.MonoDet')

def add_monodet_config(cfg: CN):
    _C = cfg

    _C.TRAINING = CN()
    _C.TRAINING.FLIP_PROB = 0.5
    _C.TRAINING.PMD_PROB = 0.5
    _C.TRAINING.POS_LABELS = []
    _C.TRAINING.MIN_VIS = 0.3
    _C.TRAINING.MIN_HEIGHT = 20

    _C.TEST.VISUALIZE = False

    _C.INPUT.CONS_SIZE = []

    _C.DATASETS.PATH = ''
    _C.DATASETS.SPLIT = ''

    # ---------------------------------------------------------------------------- #
    # att Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ROI_ATT_HEAD = CN()
    _C.MODEL.ROI_ATT_HEAD.NAME = ""
    # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
    _C.MODEL.ROI_ATT_HEAD.SMOOTH_L1_BETA = 0.0
    _C.MODEL.ROI_ATT_HEAD.POOLER_RESOLUTION = 14
    _C.MODEL.ROI_ATT_HEAD.POOLER_SAMPLING_RATIO = 0
    # Type of pooling operation applied to the incoming feature map for each RoI
    _C.MODEL.ROI_ATT_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_ATT_HEAD.NUM_FC = 0
    # Hidden layer dimension for FC layers in the RoI box head
    _C.MODEL.ROI_ATT_HEAD.FC_DIM = 1024
    _C.MODEL.ROI_ATT_HEAD.NUM_CONV = 0
    # Channel dimension for Conv layers in the RoI box head
    _C.MODEL.ROI_ATT_HEAD.CONV_DIM = 256
    # Normalization method for the convolution layers.
    # Options: "" (no norm), "GN", "SyncBN".
    _C.MODEL.ROI_ATT_HEAD.NORM = ""
    _C.MODEL.ROI_ATT_HEAD.KPT_LOSS_WEIGHT = 1.0

    # ---------------------------------------------------------------------------- #
    # dis Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ROI_DIS_HEAD = CN()
    _C.MODEL.ROI_DIS_HEAD.NAME = ""
    # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
    _C.MODEL.ROI_DIS_HEAD.SMOOTH_L1_BETA = 0.0
    _C.MODEL.ROI_DIS_HEAD.POOLER_RESOLUTION = 14
    _C.MODEL.ROI_DIS_HEAD.POOLER_SAMPLING_RATIO = 0
    # Type of pooling operation applied to the incoming feature map for each RoI
    _C.MODEL.ROI_DIS_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_DIS_HEAD.NUM_FC = 0
    # Hidden layer dimension for FC layers in the RoI box head
    _C.MODEL.ROI_DIS_HEAD.FC_DIM = 1024
    _C.MODEL.ROI_DIS_HEAD.NUM_CONV = 0
    # Channel dimension for Conv layers in the RoI box head
    _C.MODEL.ROI_DIS_HEAD.CONV_DIM = 256
    # Normalization method for the convolution layers.
    # Options: "" (no norm), "GN", "SyncBN".
    _C.MODEL.ROI_DIS_HEAD.NORM = ""
   

def set_random_seed(seed):
    logger.info("seed everything with {} and make the code deterministic".format(seed))

    # seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # make the code deterministic
    torch.backends.cudnn.benchmark = False

def load_json(json_file, image_root, cfg):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    cat_id_map = {}
    for i in range(len(cfg.TRAINING.POS_LABELS)):
        for ca in cats:
            if cfg.TRAINING.POS_LABELS[i] == ca['name']:
                cat_id_map[ca['id']] = i
    pos_samples = 0
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]

        calib = img_dict["calib"]
        P = np.zeros([4, 4], dtype=float)
        for i in range(3):
            for j in range(4):
                P[i][j] = calib[i][j]
        P[3][3] = 1
        P_inv = np.linalg.inv(P)
        record["calib"] = P
        record["calib_inv"] = P_inv
        record["image_id"] = img_dict["id"]

        bbox_dis_list = []
        for i in range(len(anno_dict_list)):
            anno = anno_dict_list[i]
            assert anno["image_id"] == record["image_id"]
            if (cats[anno["category_id"]]['name'] == 'DontCare'):
                continue
            bbox_dis_list.append([anno['bbox'], anno['location'][-1]])

        objs = []
        for i in range(len(anno_dict_list)):
            anno = anno_dict_list[i]
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == record["image_id"]

            if (cats[anno["category_id"]]['name'] not in (cfg.TRAINING.POS_LABELS + ['DontCare'])):
                continue

            ###
            count_image = np.zeros((img_dict["height"], img_dict["width"]))
            count_image[int(anno['bbox'][1]):int(anno['bbox'][3]), int(anno['bbox'][0]):int(anno['bbox'][2])] = 1
            visible_ratio = count_image.sum()
        
            for bbox_dis in bbox_dis_list:
                if bbox_dis[1] < anno['location'][-1]:
                    f_bbox = bbox_dis[0]
                    count_image[int(f_bbox[1]):int(f_bbox[3]), int(f_bbox[0]):int(f_bbox[2])] = 0
        
            visible_ratio = count_image.sum() / float(visible_ratio)
            ####
            
            point_3d = compute_box_3d(anno['dimensions'], anno['location'], anno['rotation_y'])
            point_2d = project_to_image(point_3d, record["calib"])  

            obj = {}
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            obj["bbox"] = anno['bbox']
            obj["dimensions"] = anno['dimensions']
            obj["rotation_y"] = anno['rotation_y']
            obj["alpha"] = anno['alpha']
            obj["depth"] = point_2d[-1, 2]
            obj["proj_kpts"] = point_2d[:, :2]
            cx2d = obj['proj_kpts'][-1][0]
            cy2d = obj['proj_kpts'][-1][1]
            fy = record["calib"][1, 1]
            obj["proj_h"] = fy * obj["dimensions"][0] / obj["depth"]

            assert obj["bbox"][0] >= 0 and obj["bbox"][1] >= 0 and obj["bbox"][2] <= record["width"] and obj["bbox"][3] <= record["height"]
    
            '''
            To ignore fully occluded objects during training, detectron2/data/build.py(L55), detectron2/modeling/proposal_generator/rpn.py(L292), 
            and detectron2/modeling/roi_heads/roi_heads.py(L272) have been modified.
            '''
            if (cats[anno["category_id"]]['name'] == 'DontCare') or \
            (visible_ratio < cfg.TRAINING.MIN_VIS) or \
            (not (cx2d >= 0 and cy2d >= 0 and cx2d <= record["width"] and cy2d <= record["height"])) or \
            (anno['bbox'][3] - anno['bbox'][1] < cfg.TRAINING.MIN_HEIGHT):
                obj["category_id"] = -1
            else:
                obj["category_id"] = cat_id_map[anno["category_id"]]
                pos_samples += 1   
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    logger.info("Positive labels: {}".format(cfg.TRAINING.POS_LABELS))
    logger.info("Positive samples: {}".format(pos_samples))
    return dataset_dicts

    
def register_dataset(cfg):
    split = cfg.DATASETS.SPLIT
    dataset_path = cfg.DATASETS.PATH

    train_image = os.path.join(dataset_path, split, 'train', 'image_2')
    test_image = os.path.join(dataset_path, split, 'val', 'image_2')

    train_json = os.path.join(dataset_path, split, (cfg.DATASETS.TRAIN)[0] + '.json')
    test_json = os.path.join(dataset_path, split, (cfg.DATASETS.TEST)[0] + '.json')

    DatasetCatalog.register((cfg.DATASETS.TRAIN)[0], lambda: load_json(train_json, train_image, cfg))
    DatasetCatalog.register((cfg.DATASETS.TEST)[0], lambda: load_json(test_json, test_image, cfg))


def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    
    dims = [obj["dimensions"] for obj in annos]
    dims = torch.tensor(dims, dtype=torch.float32)
    target.gt_dims = dims

    alphas = [obj["alpha"] for obj in annos]
    alphas = torch.tensor(alphas, dtype=torch.float32)
    target.gt_alphas = alphas

    proj_kpts = [obj["proj_kpts"] for obj in annos]
    proj_kpts = torch.tensor(proj_kpts, dtype=torch.float32)
    target.gt_proj_kpts = proj_kpts

    proj_hs = [obj["proj_h"] for obj in annos]
    proj_hs = torch.tensor(proj_hs, dtype=torch.float32)
    target.gt_proj_hs = proj_hs
  
    return target

class MonoDetMapper:
    def __init__(self, cfg, is_train=True):
        self.size = cfg.INPUT.CONS_SIZE
        self.flip_prob = cfg.TRAINING.FLIP_PROB
        self.pmd_prob = cfg.TRAINING.PMD_PROB
        self.is_train = is_train
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if self.is_train:
            image = utils.read_image(dataset_dict["file_name"], format="BGR")
            annos = dataset_dict.pop("annotations")
            if random.uniform(0, 1) < self.flip_prob:
                image, annos = aug_flip(image, dataset_dict["calib_inv"], annos)
            image, annos = aug_resize(image, self.size, annos)
            if random.uniform(0, 1) < self.pmd_prob:
                image = aug_pmd(image)
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
            instances = annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            return dataset_dict
        else:
            return dataset_dict

def model_pred_function(image, model, cfg):
    height, width = image.shape[:2]
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}
    predictions = model([inputs])[0]
    return predictions

def inference_on_dataset(cfg, model, data_loader, iteration):
    num_devices = get_world_size()
    logger.info("Starting inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    num_warmup = min(5, total - 1)
    
    if iteration is not None:
        iteration = (('%05d') % iteration)
    else:
        iteration = "test"

    result_dir = os.path.join(cfg.OUTPUT_DIR, 'evaluation', iteration)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    result_dir = os.path.join(cfg.OUTPUT_DIR, 'evaluation', iteration, 'data')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if cfg.TEST.VISUALIZE and iteration == "test":
        visual_dir = os.path.join(cfg.OUTPUT_DIR, 'evaluation', iteration, 'visualization')
        if not os.path.exists(visual_dir):
            os.mkdir(visual_dir)
    
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, dataset_dict in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            dataset_dict = dataset_dict[0]
            file_name = dataset_dict['file_name']
            calib = dataset_dict['calib'] 
            calib_inv = dataset_dict['calib_inv']            
            image = cv2.imread(file_name)
            image_height = image.shape[0]
            image_width = image.shape[1]
            resized_image, scale_factor = aug_resize(image, cfg.INPUT.CONS_SIZE)

            start_compute_time = time.perf_counter()
            outputs = model_pred_function(resized_image, model, cfg)
            total_compute_time += time.perf_counter() - start_compute_time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)

            pred_classes = outputs['instances'].pred_classes.cpu()
            pred_scores = outputs['instances'].scores.cpu()
            pred_boxes = outputs['instances'].pred_boxes.tensor.cpu() / scale_factor
            pred_proj_kpts = outputs['instances'].pred_proj_kpts.cpu() / scale_factor
            pred_dims = outputs['instances'].pred_dims.cpu()
            pred_yaws = outputs['instances'].pred_yaws.cpu()
            pred_Hs = outputs['instances'].pred_Hs.cpu()
            pred_hrecs = outputs['instances'].pred_hrecs.cpu() * scale_factor
            pred_hrec_uncers = outputs['instances'].pred_hrec_uncers.cpu() * scale_factor
            
            # save results
            file = open(os.path.join(result_dir, file_name.split('/')[-1][:-4] + '.txt'), 'w')
            text_to_write = ''

            if cfg.TEST.VISUALIZE and iteration == "test":
                image_2d = image.copy()
                image_bev = init_bev()

            for boxind in range(pred_boxes.shape[0]):
                cla = cfg.TRAINING.POS_LABELS[pred_classes[boxind]]

                score = pred_scores[boxind].item()

                box = pred_boxes[boxind, :]
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()

                proj_kpts = pred_proj_kpts[boxind, :, :]
                x2d = proj_kpts[-1][0].item()
                y2d = proj_kpts[-1][1].item()

                dim = pred_dims[boxind]
                h3d = dim[0].item()
                w3d = dim[1].item()
                l3d = dim[2].item()

                alpha = pred_yaws[boxind].item()

                H = pred_Hs[boxind].item()
                hrec = pred_hrecs[boxind].item()
                hrec_uncer = pred_hrec_uncers[boxind].item()
                fy = calib[1, 1]
                z2d = fy * H * hrec
                z2d_uncer = fy * H * hrec_uncer

                coord3d = calib_inv.dot(np.array([x2d * z2d, y2d * z2d, z2d, 1]))
                x3d = coord3d[0]
                y3d = coord3d[1] + h3d / 2
                z3d = coord3d[2]
                ry3d = alpha2rot(alpha, z3d, x3d)
                
                text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} ' \
                           + '{:.6f} {:.6f}\n').format(cla, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score / z2d_uncer)               

                if cfg.TEST.VISUALIZE and iteration == "test":
                    point_3d = compute_box_3d([h3d, w3d, l3d], [x3d, y3d, z3d], ry3d)
                    point_2d = project_to_image(point_3d, calib)
                    draw_image(image_2d, point_2d)
                    draw_bev(image_bev, x3d, z3d, l3d, w3d, ry3d)

            file.write(text_to_write)
            file.close()

            if cfg.TEST.VISUALIZE and iteration == "test":
                image_bev = cv2.resize(image_bev, (image_2d.shape[0], image_2d.shape[0]))
                image_visual = cv2.hconcat([image_2d, image_bev])
                cv2.imwrite(os.path.join(visual_dir, file_name.split('/')[-1]), image_visual)

            seconds_per_img = total_compute_time / iters_after_start
            # display
            if (idx + 1) % 500 == 0:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                logger.info("Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                        ))
    
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    return result_dir, iteration


def evaluate_on_dataset(cfg, result_dir, iteration):
    # evaluate
    script = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.SPLIT, 'cpp', 'evaluate_object')
    p = subprocess.Popen([script, result_dir.replace('/data', '')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    
    for lbl in cfg.TRAINING.POS_LABELS:

        lbl = lbl.lower()

        respath_2d = os.path.join(result_dir.replace('/data', ''), 'stats_{}_detection.txt'.format(lbl))
        respath_gr = os.path.join(result_dir.replace('/data', ''), 'stats_{}_detection_ground.txt'.format(lbl))
        respath_3d = os.path.join(result_dir.replace('/data', ''), 'stats_{}_detection_3d.txt'.format(lbl))

        if os.path.exists(respath_2d):
            easy, mod, hard = parse_detection_result(respath_2d)

            print_str = 'iter_{} AP40_2d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(iteration, lbl,
                                                                                                   easy, mod, hard)
            logger.info(print_str)

        if os.path.exists(respath_gr):
            easy, mod, hard = parse_detection_result(respath_gr)

            print_str = 'iter_{} AP40_gr {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(iteration, lbl,
                                                                                                   easy, mod, hard)
            logger.info(print_str)

        if os.path.exists(respath_3d):
            easy, mod, hard = parse_detection_result(respath_3d)

            print_str = 'iter_{} AP40_3d {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(iteration, lbl,
                                                                                                   easy, mod, hard)
            logger.info(print_str)


def parse_detection_result(respath):
    text_file = open(respath, 'r')

    acc = np.zeros([3, 41], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall('([\d]+\.?[\d]*)', line)

        for i, num in enumerate(parsed):
            acc[lind, i] = float(num)

        lind += 1

    text_file.close()

    easy = np.mean(acc[0, 1:41:1])
    mod = np.mean(acc[1, 1:41:1])
    hard = np.mean(acc[2, 1:41:1])

    return easy, mod, hard
    
@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
