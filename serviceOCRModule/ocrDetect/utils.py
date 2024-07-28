#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : PaddleOCR2openvino
# @Time : 2024/7/28 上午10:17

import cv2
import math
import numpy as np

def resize_image(img,short_side=736):
    height, width, _ = img.shape
    if height < width:
        new_height = short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img  


def post_img(img, short_side=736):
    img = resize_image(img, short_side)
    img = to_tensor_numpy(img)
    img = normalize_numpy(img)
    img = np.expand_dims(img, axis=0)
    return img


def to_tensor_numpy(img):
    # 将图像转换为张量
    img_tensor = np.transpose(img, (2, 0, 1))
    img_tensor = img_tensor.astype(np.float32) / 255.0
    return img_tensor

def normalize_numpy(img):
    # 归一化处理
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img - mean[:, None, None]) / std[:, None, None]
    return img_normalized