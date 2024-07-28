#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : PaddleOCR2openvino
# @Time : 2024/7/28 上午10:17

cls_model_file = "./serviceOCRModule/ocrCls/openvino_dir/cls.xml"
detect_model_file = "./serviceOCRModule/ocrDetect/openvino_dir/det.xml"
recog_model_file = "./serviceOCRModule/ocrRecog/openvino_dir/rec.xml"

recog_keys_file = "./serviceOCRModule/ocrRecog/ppocr_keys_v1.txt"

detect_params = {}
detect_params['thresh'] = 0.3
detect_params['box_thresh'] = 0.6
detect_params['max_candidates'] = 1000 
detect_params['is_poly'] = False
detect_params['unclip_ratio'] = 1.5
detect_params['min_size'] = 5