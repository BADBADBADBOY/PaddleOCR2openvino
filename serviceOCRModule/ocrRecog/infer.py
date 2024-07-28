#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @Project : PaddleOCR2openvino
# @Time : 2024/7/28 上午10:17

import cv2
import os
import sys
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from ocrRecog.utils import resize_norm_img,CTCLabelDecode

class ppRecogOpenvino(object):
    def __init__(self,model_path,keys_file,use_space_char=False):
        super(ppRecogOpenvino,self).__init__()
        from openvino.runtime import Core, AsyncInferQueue
        ie = Core()
        model_ir = ie.read_model(model=model_path)
        compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU", config={"PERFORMANCE_HINT":"LATENCY","CPU_RUNTIME_CACHE_CAPACITY":"0"})
        self.infer_request = compiled_model_ir.create_infer_request()
        self.decode_ctc = CTCLabelDecode(keys_file,use_space_char=use_space_char)
        
    def recog_img(self,img):
        max_wh_ratio = img.shape[1]/img.shape[0]
        img = resize_norm_img(img,max_wh_ratio)
        img = img[np.newaxis, :]
        self.infer_request.infer([img])
        out = self.infer_request.get_output_tensor(0).data
        text = self.decode_ctc(out)
        return text[0]