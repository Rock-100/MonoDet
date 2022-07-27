import numpy as np
import math
import os
import sys
import random
import cv2
cv2.setNumThreads(0)

from lib.transformation import (
    alpha2rot,
    rot2alpha,
)

def aug_pmd(img, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """
    contrast_lower, contrast_upper = contrast_range
    saturation_lower, saturation_upper = saturation_range

    img = img.copy().astype(np.float32)
    assert img.dtype == np.float32, \
        'PhotoMetricDistortion needs the input image of dtype np.float32,'\
        ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
    # random brightness
    if random.randint(0, 1):
        delta = random.uniform(-brightness_delta, brightness_delta)
        img += delta

    # mode == 0 --> do random contrast first
    # mode == 1 --> do random contrast last
    mode = random.randint(0, 1)
    if mode == 1:
        if random.randint(0, 1):
            alpha = random.uniform(contrast_lower, contrast_upper)
            img *= alpha

    # convert color from BGR to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # random saturation
    if random.randint(0, 1):
        img[..., 1] *= random.uniform(saturation_lower, saturation_upper)

    # random hue
    if random.randint(0, 1):
        img[..., 0] += random.uniform(-hue_delta, hue_delta)
        img[..., 0][img[..., 0] > 360] -= 360
        img[..., 0][img[..., 0] < 0] += 360

    # convert color from HSV to BGR
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # random contrast
    if mode == 0:
        if random.randint(0, 1):
            alpha = random.uniform(contrast_lower, contrast_upper)
            img *= alpha

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def aug_resize(image, input_size, annos=None):
    scale_factor = float(input_size[0]) / image.shape[0]
    h = np.round(image.shape[0] * scale_factor).astype(int)
    w = np.round(image.shape[1] * scale_factor).astype(int)
    # resize
    image = cv2.resize(image, (w, h))
    assert image.shape[0] == input_size[0] and image.shape[1] <= input_size[1]
   
    if annos is not None:
        # pad out
        if image.shape[1] < input_size[1]:
            padW = input_size[1] - image.shape[1]
            image = np.pad(image, [(0, 0), (0, padW), (0, 0)], 'constant')
        
        for i in range(len(annos)):
            for j in range(4):
                annos[i]['bbox'][j] *= scale_factor
            annos[i]['proj_kpts'] = [[p[0] * scale_factor, p[1] * scale_factor] for p in annos[i]['proj_kpts']]
            annos[i]['proj_h'] = annos[i]['proj_h'] * scale_factor
        return image, annos
    else:
        return image, scale_factor

def aug_flip(image, calib_inv, annos):
    image = cv2.flip(image, 1)
    width = image.shape[1]

    for i in range(len(annos)):
        for j in [0, 2]:
            annos[i]['bbox'][j] = width -1 - annos[i]['bbox'][j]
        annos[i]['bbox'][0], annos[i]['bbox'][2] = annos[i]['bbox'][2], annos[i]['bbox'][0]
        annos[i]['proj_kpts'] = [[width - 1 - p[0], p[1]] for p in annos[i]['proj_kpts']]
        annos[i]['proj_kpts'][0], annos[i]['proj_kpts'][1] = annos[i]['proj_kpts'][1], annos[i]['proj_kpts'][0]
        annos[i]['proj_kpts'][2], annos[i]['proj_kpts'][3] = annos[i]['proj_kpts'][3], annos[i]['proj_kpts'][2]
        annos[i]['proj_kpts'][4], annos[i]['proj_kpts'][5] = annos[i]['proj_kpts'][5], annos[i]['proj_kpts'][4]
        annos[i]['proj_kpts'][6], annos[i]['proj_kpts'][7] = annos[i]['proj_kpts'][7], annos[i]['proj_kpts'][6]
        
        rotY = annos[i]['rotation_y']
        rotY = (-math.pi - rotY) if rotY < 0 else (math.pi - rotY)
        while rotY > math.pi: rotY -= math.pi * 2
        while rotY < (-math.pi): rotY += math.pi * 2
            
        cx2d = annos[i]['proj_kpts'][-1][0]
        cy2d = annos[i]['proj_kpts'][-1][1]
        cz2d = annos[i]['depth']

        coord3d = calib_inv.dot(np.array([cx2d * cz2d, cy2d * cz2d, cz2d, 1]))

        alpha = rot2alpha(rotY, coord3d[2], coord3d[0])

        annos[i]['rotation_y'] = rotY
        annos[i]['alpha'] = alpha

    return image, annos
