import cv2
import math
import random
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from serviceOCRModule.ocrCls.infer import ppClsOpenvino
from serviceOCRModule.ocrDetect.infer import ppDetectOpenvino
from serviceOCRModule.ocrRecog.infer import ppRecogOpenvino
from serviceOCRModule.service_config import cls_model_file,detect_model_file,recog_model_file,recog_keys_file,detect_params

def create_font(txt, sz, font_path="./simfang.ttf"):
    font_size = int(sz[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    length = font.getlength(txt)
    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font

def draw_box_txt_fine(img_size, box, txt, font_path="./simfang.ttf"):
    box_height = int(
        math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2))
    box_width = int(
        math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2))

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new('RGB', (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new('RGB', (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255))
    return img_right_text

def draw_ocr_box_txt(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    img_show = np.array(img_show)
    for bbox in boxes:
        bbox = bbox.reshape(-1, 2).astype(np.int32)
        img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
    return img_show

## 计算欧式距离
def cal_distance(coord1,coord2):
    return math.sqrt((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2)

## 得到文字的长和宽
def cal_width_height(bbox):
    width = cal_distance((bbox[0],bbox[1]),(bbox[2],bbox[3]))
    height = cal_distance((bbox[2],bbox[3]),(bbox[4],bbox[5]))
    return int(width),int(height)

def get_perspective_image(image,bbox):
    width,height = cal_width_height(bbox)
    if height>width:
        pts1 = np.float32([[0,0],[height,0],[height,width],[0,width]])
        pts2 = np.float32(np.array([bbox[2],bbox[3],bbox[4],bbox[5],bbox[6],bbox[7],bbox[0],bbox[1]]).reshape(4,2))
        width,height = height,width
    else:
        pts1 = np.float32([[0,0],[width,0],[width,height],[0,height]])
        pts2 = np.float32(bbox.reshape(4,2))
    M = cv2.getPerspectiveTransform(pts2,pts1)
    dst = cv2.warpPerspective(image,M,(width,height))
    return dst

def testone(img,short_size = 736):
    bbox_batch = ppdetect_bin.det_img(img,short_size)
    show_boxes,show_txts,show_scores =[],[],[]
    for box in bbox_batch:
        cut_img = get_perspective_image(img,box.reshape(-1))
        cut_img,cls,cls_conf = ppcls_bin.cls_img(cut_img)
        recog_text = pprecog_bin.recog_img(cut_img)
        show_boxes.append(box.reshape(4,2))
        show_txts.append(recog_text[0])
        show_scores.append(recog_text[1])
    return show_boxes,show_txts,show_scores


ppcls_bin = ppClsOpenvino(cls_model_file)
ppdetect_bin = ppDetectOpenvino(detect_model_file,detect_params)
pprecog_bin = ppRecogOpenvino(recog_model_file,recog_keys_file)

img = cv2.imread("./test_img/test1.jpg")

for i in range(1):   
    t = time.time()
    show_boxes,show_txts,show_scores = testone(img)
    print("time:{}".format(time.time() - t))
for txt,score in zip(show_txts,show_scores):
    print(txt,score)
img = Image.fromarray(img)
img_show = draw_ocr_box_txt(img,show_boxes,show_txts,show_scores,drop_score=0.5)
cv2.imwrite('show_service.jpg',img_show)


