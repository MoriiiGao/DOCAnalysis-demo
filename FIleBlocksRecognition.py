import base64
import json
import math
from typing import List
#
import fitz
import os
import time
from pathlib import Path
import random

import numpy as np
import torch
import cv2

from paddleocr.tools.infer.utility import create_font
from utils.plots import Annotator, colors
FILE = Path(__file__).resolve()
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
basepath = os.path.dirname(__file__)
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

def make_output(path: List):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
    return True

def draw_ocr_box_txt(image,
                     dst,
                     boxes,
                     txts=None,
                     scores=None,
                     drop_score=0.5,
                     font_path="/home/hello/gzc/3.project/yolov5-master/fonts/simfang.ttf"):
    # image = Image.open(image)
    # # h, w = image.height, image.width #93 241
    h, w, c = dst.shape[:] #171 241
    # img_left = image.copy()
    img_left = Image.fromarray(dst)
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
        # draw_left.polygon(box, fill=color)#box必须是两个或多个位置坐标
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    # img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)

def draw_box_txt_fine(img_size, box, txt, font_path="./doc/fonts/simfang.ttf"):
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

def expand(img):
    h, w, c = img.shape[:]
    border = [0, 0]
    transform_size = 320  # 图片增加边框到320大小
    if w < transform_size or h < transform_size:
        if h < transform_size:
            border[0] = (transform_size - h) / 2.0
        if w < transform_size:
            border[0] = (transform_size - w) / 2.0

        # top, buttom, left, right对应边界的像素数目
        img = cv2.copyMakeBorder(img, int(border[0]), int(border[0]), int(border[1]), int(border[1]),
                                 cv2.BORDER_CONSTANT,
                                 value=[215, 215, 215])

        # image_file = "result/pix_expend/" + self.otherStyleTime + '.png'
        # cv2.imwrite(image_file, img)

    return img

def OCR(img):
    dst = cv2.imread(img)
    # dst = Image.open(img)
    dst = expand(dst)
    result = ocr.ocr(dst, cls=True)
    # boxes = [line1 for line in result for line1 in line[0][0]]
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    # im_show = draw_ocr(dst, boxes, txts, scores)
    im_show = draw_ocr_box_txt(img, dst, boxes, txts, scores)
    im_show = Image.fromarray(im_show)

    SavePath ='ocr.png'
    im_show.save(SavePath)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf8')
        return json.JSONEncoder.default(self, obj)

class BlocksRecognition():
    """
        /home/hello/gzc/3.project/yolov5-master/runs/train/exp3/weights
    /home/ubuntu18/gzc/DigitalDocumentAnalysisSystem/yolov5-master/runs/train/exp3/weights/best.pt
    """
    def __init__(self, weights='/home/hello/gzc/3.project/yolov5-master/runs/train/exp3/weights/best.pt',  # model.pt path(s)
            data='data/coco.yaml',  # dataset.yaml path
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            imgsz=(640, 640),
            agnostic_nms=False,  # class-agnostic NMS
            line_thickness=1,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            dnn=False):
        super(BlocksRecognition, self).__init__()

        self.url = 'http://172.16.16.112:8888/'
        self.font = "/home/hello/gzc/3.project/yolov5-master/fonts/simfang.ttf"
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.half = False
        self.model.model.float()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det

        self.line_thickness = line_thickness
        self.view_img = view_img
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

        self.time = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.upload_file_path = './static/AnalysisSystem/BlocksRecognition/upload_file/' + self.time + '/'
        self.upload_file_to_img_path = './static/AnalysisSystem/BlocksRecognition/ch_pdf/pdf_to_img/' + self.time + '/'
        self.layout_img = './static/AnalysisSystem/BlocksRecognition/layout/' + self.time + '/'

        self.recog_fig_path = './static/AnalysisSystem/BlocksRecognition/fig/' + self.time + '/'
        self.recog_tab_path = './static/AnalysisSystem/BlocksRecognition/tab/' + self.time + '/'
        self.recog_equa_path = './static/AnalysisSystem/BlocksRecognition/equa/' + self.time + '/'
        self.recog_text_path = './static/AnalysisSystem/BlocksRecognition/text/' + self.time + '/'
        self.recog_title_path = './static/AnalysisSystem/BlocksRecognition/title/' + self.time + '/'
        self.recog_footer_path = './static/AnalysisSystem/BlocksRecognition/footer/' + self.time + '/'
        self.recog_header_path = './static/AnalysisSystem/BlocksRecognition/header/' + self.time + '/'
        self.recog_ref_path = './static/AnalysisSystem/BlocksRecognition/ref/' + self.time + '/'
        self.OCR_res = './static/AnalysisSystem/BlocksRecognition/OCR/' + self.time + '/'
        self.detect_path = './static/AnalysisSystem/BlocksRecognition/detect/' + self.time + '/'

        make_output([self.upload_file_path, self.upload_file_to_img_path, self.layout_img, self.recog_fig_path,
                     self.recog_tab_path, self.recog_equa_path, self.recog_text_path, self.recog_title_path,
                     self.recog_footer_path, self.recog_header_path, self.recog_ref_path, self.OCR_res, self.detect_path])

    def border_extension(self, img):
        """
        问题：paddleocr无法精准识别小图和长图
        方法：对图片进行预处理 增加图片的大小 给图片加一定大小的框
        对裁剪出来的小图和长图 在边界填充一定数目的像素
        """

        h, w, c = img.shape[:]
        border = [0, 0]
        transform_size = 320  # 图片增加边框到320大小
        if w < transform_size or h < transform_size:
            if h < transform_size:
                border[0] = (transform_size - h) / 2.0
            if w < transform_size:
                border[0] = (transform_size - w) / 2.0

            # top, buttom, left, right对应边界的像素数目
            img = cv2.copyMakeBorder(img, int(border[0]), int(border[0]), int(border[1]), int(border[1]),
                                     cv2.BORDER_CONSTANT,
                                     value=[215, 215, 215])

            # image_file = "result/pix_expend/" + self.otherStyleTime + '.png'
            # cv2.imwrite(image_file, img)

        return img

    def draw_box_txt_fine(self,img_size, box, txt, font_path="./doc/fonts/simfang.ttf"):

        box_height = int(
            math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2))
        box_width = int(
            math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))

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


    def draw_ocr_box_txt(self, dst, boxes, txts=None, scores=None,
                          drop_score=0.5, font_path="/home/hello/gzc/3.project/yolov5-master/fonts/simfang.ttf"):

        # image = Image.open(image)
        # # h, w = image.height, image.width #93 241
        h, w, c = dst.shape[:]  # 171 241
        # img_left = image.copy()
        img_left = Image.fromarray(dst)
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
            # draw_left.polygon(box, fill=color)#box必须是两个或多个位置坐标
            img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_right_text, [pts], True, color, 1)
            img_right = cv2.bitwise_and(img_right, img_right_text)
        # img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
        return np.array(img_show)

    def OCR(self, loc_img, dst_img, idx, pgnum):
        # img = cv2.imread(loc_img)
        dst_img = self.border_extension(dst_img)
        result = ocr.ocr(dst_img, cls=True)
        # boxes = [line1 for line in result for line1 in line[0][0]]
        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]
        # im_show = draw_ocr(dst_img, boxes, txts, scores)
        im_show = self.draw_ocr_box_txt(dst_img, boxes, txts, scores)
        im_show = Image.fromarray(im_show)

        SavePath = self.OCR_res + '{}_{}.png'.format(pgnum, idx)
        im_show.save(SavePath)
        OCRURLPath = self.url + self.OCR_res.split('.')[-1] + '{}_{}.png'.format(pgnum, idx)
        return OCRURLPath


    def CutImage(self, axis, Image, label, pgnum, idx):
        """
        裁切文档元素
        axis：元素位置((x1, y1), (x2, y2))
        Image：原始图像
        Text, Title, Figure, Table, Header, Footer, Reference, Equation
        """
        image = cv2.imread(Image)
        if label == 'Text':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_text_path + '{}_{}.png'.format(pgnum, idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_text_path.split('.')[-1] + '{}_{}.png'.format(pgnum, idx)
            # img_bytes = dst_img.tobytes()
            # img_base64 = base64.b64encode(img_bytes)
        elif label == 'Title':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_title_path + '{}_{}.png'.format(pgnum, idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_title_path.split('.')[-1] + '{}_{}.png'.format(pgnum, idx)
        elif label == 'Table':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_tab_path + '{}_{}.png'.format(pgnum, idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_tab_path.split('.')[-1] + '{}_{}.png'.format(pgnum, idx)
        elif label == 'Header':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_header_path + '{}_{}.png'.format(pgnum, idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_header_path.split('.')[-1] + '{}_{}.png'.format(pgnum, idx)
        elif label == 'Footer':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_footer_path + '{}_{}.png'.format(pgnum, idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_footer_path.split('.')[-1] + '{}_{}.png'.format(pgnum, idx)
        elif label == 'Reference':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_ref_path + '{}_{}.png'.format(pgnum, idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_ref_path.split('.')[-1] + '{}_{}.png'.format(pgnum, idx)
        else:
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_equa_path + '{}_{}.png'.format(pgnum, idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_equa_path.split('.')[-1] + '{}_{}.png'.format(pgnum, idx)
        return img_path, img_url, dst_img

    def layout_analysis(self, img_path, pgnum):
        source = str(img_path)
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

            im0 = im0s.copy()

            #interface
            pred = self.model(im, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)

            DATA = []
            for i, det in enumerate(pred): #per image
                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                        data_tmp = {}
                        if self.view_img:  # Add bbox to image
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            label = self.names[int(cls)]
                            axis = ((x1, y1), (x2, y2))
                            str_axis = ((str(x1), str(y1)), (str(x2), str(y2)))
                            annotator.box_label(xyxy, label, color=colors(int(cls), True))

                            #ocr识别
                            if label in ['Text', 'Title', 'Table', 'Reference']:
                                #像素裁切
                                loc_img, url_img, dst_img = self.CutImage(axis, source, label, pgnum, idx)
                                #OCR识别
                                OCRURLPath = self.OCR(loc_img, dst_img, idx, pgnum)
                                data_tmp['label'] = label
                                data_tmp['img'] = url_img
                                data_tmp['ocr'] = OCRURLPath
                                DATA.append(data_tmp)

                im0 = annotator.result()
                cv2.imwrite(self.detect_path + '/' + str(pgnum) + '.png', im0)
            return DATA


    def main(self, PDFPath):

        PDFname = PDFPath.split('/')[-1].split('.')[0]
        PDFDOC = fitz.open(PDFPath)
        DATA = list()
        for pgnum in range(PDFDOC.pageCount):
            data_tmp = {}
            page = PDFDOC[pgnum]
            rotate = int(0)
            zoom_x = 1.0
            zoom_y = 1.0
            mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pix = page.getPixmap(matrix=mat)

            IMGPath = self.upload_file_to_img_path + '/' + PDFname + '_' + 'images_%s.png' % pgnum
            pix.writePNG(IMGPath)
            page_res = self.layout_analysis(IMGPath, pgnum)
            data_tmp['page_num'] = pgnum
            data_tmp['page_res'] = page_res
            DATA.append(data_tmp)
        data = {'code': 200, 'msg': 'success', 'PDFFile': PDFPath, 'data': DATA}
        # print(data)
        return data

if __name__ == '__main__':
    # img = 'test/pdf2html/small.png'
    # OCR(img)
    # img = 'test/pdf2html/test_3_images_0.png'
    # OCR(img)
    pdf_path = 'test/pdf2html/test_3.pdf'
    BR = BlocksRecognition()
    BR.main(pdf_path)