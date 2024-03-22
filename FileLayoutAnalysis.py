import base64
import json
import shutil
from typing import List
#
import fitz
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import tqdm

from utils.plots import Annotator, colors
FILE = Path(__file__).resolve()
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from flask import Flask, request, jsonify

basepath = os.path.dirname(__file__)

def make_output(path: List):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
    return True

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf8')
        return json.JSONEncoder.default(self, obj)

class LayoutAnalysis():
    """
        /home/hello/gzc/3.project/yolov5-master/runs/train/exp3/weights
    /home/ubuntu18/gzc/DigitalDocumentAnalysisSystem/yolov5-master/runs/train/exp3/weights/best.pt
    """
    def __init__(self, weights='/home/hello/gzc/3.project/yolov5-master/runs/train/exp3/weights/best.pt',  # model.pt path(s)
            data='data/coco.yaml',  # dataset.yaml path                     数据集配置文件路径
            conf_thres=0.25,  # confidence threshol                         置信度的阈值 超过这个阈值的预测框就会被预测出来。
            iou_thres=0.45,  # NMS IOU threshold                            iou阈值
            max_det=1000,  # maximum detections per image                   每张图最大检测数量
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results                                  检测的时候是否实时把检测结果显示出来
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            imgsz=(640, 640),
            agnostic_nms=False,  # class-agnostic NMS                       跨类别NMS
            line_thickness=1,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            dnn=False):
        super(LayoutAnalysis, self).__init__()

        self.url = 'http://172.16.16.112:8888/'
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
        self.upload_file_path = './static/AnalysisSystem/LayoutAnalysis/upload_file/' + self.time + '/'
        self.upload_file_to_img_path = './static/AnalysisSystem/LayoutAnalysis/ch_pdf/pdf_to_img/' + self.time + '/'
        self.layout_img = './static/AnalysisSystem/LayoutAnalysis/layout/' + self.time + '/'

        self.recog_fig_path = './static/AnalysisSystem/LayoutAnalysis/fig/' + self.time + '/'
        self.recog_tab_path = './static/AnalysisSystem/LayoutAnalysis/tab/' + self.time + '/'
        self.recog_equa_path = './static/AnalysisSystem/LayoutAnalysis/equa/' + self.time + '/'
        self.recog_text_path = './static/AnalysisSystem/LayoutAnalysis/text/' + self.time + '/'
        self.recog_title_path = './static/AnalysisSystem/LayoutAnalysis/title/' + self.time + '/'
        self.recog_footer_path = './static/AnalysisSystem/LayoutAnalysis/footer/' + self.time + '/'
        self.recog_header_path = './static/AnalysisSystem/LayoutAnalysis/header/' + self.time + '/'
        self.recog_ref_path = './static/AnalysisSystem/LayoutAnalysis/ref/' + self.time + '/'
        self.detect_path = './static/AnalysisSystem/LayoutAnalysis/detect/' + self.time + '/'

        make_output([self.upload_file_path, self.upload_file_to_img_path, self.layout_img, self.recog_fig_path,
                     self.recog_tab_path, self.recog_equa_path, self.recog_text_path, self.recog_title_path,
                     self.recog_footer_path, self.recog_header_path, self.recog_ref_path, self.detect_path])

    def CutImage(self, axis, Image, label, idx):
        """
        裁切文档元素
        axis：元素位置((x1, y1), (x2, y2))
        Image：原始图像
        Text, Title, Figure, Table, Header, Footer, Reference, Equation
        """
        image = cv2.imread(Image)
        if label == 'Text':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_text_path + '{}.png'.format(idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_text_path.split('.')[-1] + '{}.png'.format(idx)
            # img_bytes = dst_img.tobytes()
            # img_base64 = base64.b64encode(img_bytes)
        elif label == 'Title':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_title_path + '{}.png'.format(idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_title_path.split('.')[-1] + '{}.png'.format(idx)
        elif label == 'Table':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_tab_path + '{}.png'.format(idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_tab_path.split('.')[-1] + '{}.png'.format(idx)
        elif label == 'Header':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_header_path + '{}.png'.format(idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_header_path.split('.')[-1] + '{}.png'.format(idx)
        elif label == 'Footer':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_footer_path + '{}.png'.format(idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_footer_path.split('.')[-1] + '{}.png'.format(idx)
        elif label == 'Reference':
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_ref_path + '{}.png'.format(idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_ref_path.split('.')[-1] + '{}.png'.format(idx)
        else:
            dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
            img_path = self.recog_equa_path + '{}.png'.format(idx)
            cv2.imwrite(img_path, dst_img)
            img_url = self.url + self.recog_equa_path.split('.')[-1] + '{}.png'.format(idx)
        return img_url

    def np_2_base64(self, PixMat):
        img_bytes = PixMat.tobytes()
        img_base64 = base64.b64encode(img_bytes)
        return img_base64

    def layout_analysis(self, img_path, pgnum):
        """
        return ：Det_img,
                {
                    'id':
                    'label':base64,
                    'axis':((), ())
                }
        """
        source = str(img_path)
        name = source.split('/')[-1].split('.')[1]
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
            #预测
            IMG_DATA = list()
            AXIS_DATA = list()
            for i, det in enumerate(pred): #per image
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                        img_data_tmp = {}
                        axis_data_tmp = {}
                        if self.view_img:  # Add bbox to image
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            label = self.names[int(cls)]
                            axis = ((x1, y1), (x2, y2))
                            str_axis = ((str(x1), str(y1)), (str(x2), str(y2)))
                            if label in ['Text', 'Title', 'Figure', 'Table']:
                                annotator.box_label(xyxy, label, color=colors(int(cls), True))
                            Element_img = self.CutImage(axis, source, label, idx)
                            img_data_tmp['id'] = idx
                            img_data_tmp['label'] = label
                            img_data_tmp['img'] = Element_img
                            axis_data_tmp['axis'] = str_axis
                            IMG_DATA.append(img_data_tmp)
                            AXIS_DATA.append(axis_data_tmp)
                    im0 = annotator.result()
                    cv2.imwrite(self.detect_path + '/' + str(pgnum) + '.png', im0)
                    det_img = self.url + self.detect_path.split('.')[-1] + '{}.png'.format(pgnum)
                    # im0_base64 = self.np_2_base64(im0)

            return {'det_img': det_img, 'img_data': IMG_DATA, 'axis_data': AXIS_DATA}

    def pdf2img(self, PDFPath):

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
            data_tmp['page_num'] = str(pgnum)
            data_tmp['page_res'] = page_res
            DATA.append(data_tmp)

        # data1 = json.dumps({'code': 200, 'msg': 'success', 'PDFFile': PDFPath, 'data': DATA})
        # data = json.dumps({'code': 200, 'msg': 'success', 'PDFFile': PDFPath, 'data': DATA},
        #                     # cls=MyEncoder,
        #                     indent=4,
        #                     ensure_ascii=False)
        data = {'code': 200, 'msg': 'success', 'PDFFile': PDFPath, 'data': DATA}
        # with open('LayoutAnlysis.json', 'w') as json_file:
        #     json_file.write(data)
        print(data)
        return data


def img_test():
    LA = LayoutAnalysis()

    path = '/home/hello/gzc/3.project/yolov5-master/pdftest/imgtest/'
    # imgpath = 'pdftest/imgtest/'
    imgpath = '/data/gzc/yolo_publaynet/train/images/'
    file_list = os.listdir(imgpath)
    for pgnum in tqdm(range(len(file_list))):
        file = imgpath + file_list[pgnum]
        LA.layout_analysis(file, pgnum)


if __name__ == "__main__":
    # pdf = 'pdftest/en3.pdf'
    # pdf_path = 'pdftest/iccv2021_video_matting/iccv2021_video_matting_3.pdf'
    # pdf_path = 'test/pdf2html/test_3.pdf'
    # LA = LayoutAnalysis()
    # LA.pdf2img(pdf_path)

    img_test()