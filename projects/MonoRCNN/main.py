#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from lib.train_test import (
    add_monodet_config, 
    register_dataset, 
    MonoDetMapper,
    inference_on_dataset,
    evaluate_on_dataset,
    set_random_seed,
)

from lib.all_roi_heads import ALLROIHeads
from lib.att_head import ATTHead
from lib.dis_head import DISHead

logger = logging.getLogger("detectron2")
monodet_mapper_train = None
monodet_mapper_test = None

def do_test(cfg, model, iteration=None):
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=monodet_mapper_test)
        result_dir, iteration = inference_on_dataset(cfg, model, data_loader, iteration)
        comm.synchronize()
        if comm.is_main_process():
            evaluate_on_dataset(cfg, result_dir, iteration)
            logger.info("Evaluation for {} is completed.".format(dataset_name))
            
def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, os.path.join(cfg.OUTPUT_DIR, 'model'), optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter + 10
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg, mapper=monodet_mapper_train)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (iteration % 100 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration - 1)
            if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                do_test(cfg, model, iteration - 1)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()
            

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_monodet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def main(args):
    cfg = setup(args)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'model')):
        os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'model'))
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'evaluation')):
        os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'evaluation'))
    register_dataset(cfg)
    set_random_seed(cfg.SEED)
    global monodet_mapper_train, monodet_mapper_test
    monodet_mapper_train = MonoDetMapper(cfg, is_train=True)
    monodet_mapper_test = MonoDetMapper(cfg, is_train=False)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=os.path.join(cfg.OUTPUT_DIR, 'model')).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        do_test(cfg, model)
        return

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    do_train(cfg, model, resume=args.resume)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
