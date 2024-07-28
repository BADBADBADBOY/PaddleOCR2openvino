import math
import numpy as np
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from ocrDetect.postprocess import DBPostProcess
from ocrDetect.utils import post_img


    
class ppDetectOpenvino(object):
    def __init__(self,model_file,params):
        super(ppDetectOpenvino,self).__init__()
        from openvino.runtime import Core, AsyncInferQueue
        ie = Core()
        model_ir = ie.read_model(model=model_file)
        compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU", config={"PERFORMANCE_HINT":"LATENCY","CPU_RUNTIME_CACHE_CAPACITY":"0"})
        self.infer_request = compiled_model_ir.create_infer_request()
        
        self.dbprocess = DBPostProcess(params)
    
    def det_img(self,img_ori,shortest=736):
        img = post_img(img_ori,shortest)
        self.infer_request.infer([img])
        out = self.infer_request.get_output_tensor(0).data
        scale = (img_ori.shape[1] * 1.0 / out.shape[3], img_ori.shape[0] * 1.0 / out.shape[2])       
        bbox_batch,score_batch = self.dbprocess(out,[scale])
        return bbox_batch[0]
    