python3 tools/infer/predict_system.py --image_dir="./test_img/test1.jpg" --det_model_dir="./trained_model/ch_PP-OCRv4_det_infer/det.onnx" --cls_model_dir="./trained_model/ch_ppocr_mobile_v2.0_cls_infer/cls.onnx" --rec_model_dir="./trained_model/ch_PP-OCRv4_rec_infer/rec.onnx" --use_angle_cls=true --use_onnx True