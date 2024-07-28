import cv2
import numpy as np
import math
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from ocrCls.utils import resize_norm_img

class ppClsOpenvino(object):
    def __init__(self,model_path):
        super(ppClsOpenvino,self).__init__()
        
        from openvino.runtime import Core, AsyncInferQueue
        ie = Core()
        model_ir = ie.read_model(model=model_path)
        compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU", config={"PERFORMANCE_HINT":"LATENCY","CPU_RUNTIME_CACHE_CAPACITY":"0"})
        self.infer_request = compiled_model_ir.create_infer_request()
        
        self.angles = ['0','180']
    
    def cls_img(self,img):
        img_ori = img.copy()
        img = resize_norm_img(img)
        img = img[np.newaxis, :]
        self.infer_request.infer([img])
        out = self.infer_request.get_output_tensor(0).data
        index = out[0].argmax().item()
        if index==1:
            img_ori = cv2.rotate(img_ori,1)
        return img_ori,self.angles[index],out[0][index].item()