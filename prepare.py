import sys
import os
import shutil

def remove_pycache(directory):
    for root, dirs, files in os.walk(directory):
        if '__pycache__' in dirs:
            shutil.rmtree(os.path.join(root, '__pycache__'))
        if '.ipynb_checkpoints' in dirs:
            shutil.rmtree(os.path.join(root, '.ipynb_checkpoints'))

if sys.argv[1] == 'make':
    script = """
    mkdir ./trained_model/\n
    cd ./trained_model/\n
    wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar\n
    wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar\n
    wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar\n
    tar -xvf ch_PP-OCRv4_det_infer.tar\n
    tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar\n
    tar -xvf ch_PP-OCRv4_rec_infer.tar\n
    rm -r *.tar\n
    paddle2onnx  --model_dir ./ch_ppocr_mobile_v2.0_cls_infer/ --model_filename inference.pdmodel --params_file inference.pdiparams --save_file ./ch_ppocr_mobile_v2.0_cls_infer/cls.onnx --opset_version 11 --enable_dev_version True\n
    paddle2onnx --model_dir ./ch_PP-OCRv4_rec_infer/ --model_filename inference.pdmodel --params_file inference.pdiparams --save_file ./ch_PP-OCRv4_rec_infer/rec.onnx --opset_version 11 --enable_dev_version True\n
    paddle2onnx --model_dir ./ch_PP-OCRv4_det_infer/ --model_filename inference.pdmodel --params_file inference.pdiparams --save_file ./ch_PP-OCRv4_det_infer/det.onnx --opset_version 11 --enable_dev_version True\n
    mkdir ./ch_ppocr_mobile_v2.0_cls_infer/openvino/\n
    mkdir ./ch_PP-OCRv4_rec_infer/openvino/\n
    mkdir ./ch_PP-OCRv4_det_infer/openvino/\n
    mo --input_model=./ch_ppocr_mobile_v2.0_cls_infer/cls.onnx  --output_dir=./ch_ppocr_mobile_v2.0_cls_infer/openvino  --model_name="cls" --input_shape=[-1,3,-1,-1]\n
    mo --input_model=./ch_PP-OCRv4_det_infer/det.onnx  --output_dir=./ch_PP-OCRv4_det_infer/openvino  --model_name="det" --input_shape=[-1,3,-1,-1]\n
    mo --input_model=./ch_PP-OCRv4_rec_infer/rec.onnx  --output_dir=./ch_PP-OCRv4_rec_infer/openvino  --model_name="rec" --input_shape=[-1,3,-1,-1]\n
    mkdir ../serviceOCRModule/ocrRecog/openvino_dir/\n
    mkdir ../serviceOCRModule/ocrDetect/openvino_dir/\n
    mkdir ../serviceOCRModule/ocrCls/openvino_dir/\n
    cp ./ch_PP-OCRv4_rec_infer/openvino/* ../serviceOCRModule/ocrRecog/openvino_dir/\n
    cp ./ch_PP-OCRv4_det_infer/openvino/* ../serviceOCRModule/ocrDetect/openvino_dir/\n
    cp ./ch_ppocr_mobile_v2.0_cls_infer/openvino/* ../serviceOCRModule/ocrCls/openvino_dir/
    """
    os.system(script)
elif sys.argv[1] == 'clear':
    script = """rm -r trained_model inference_results show_service.jpg\n
    rm -r serviceOCRModule/ocrCls/openvino_dir/\n
    rm -r serviceOCRModule/ocrDetect/openvino_dir/\n
    rm -r serviceOCRModule/ocrRecog/openvino_dir/"""
    os.system(script)
    remove_pycache('./')
else:
    assert 1!=1,"only support 'make' or 'clear'"