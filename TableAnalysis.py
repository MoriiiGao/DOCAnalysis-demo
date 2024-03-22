import base64
import json
import sys
from typing import List
import fitz
import os
import time
from pathlib import Path

import numpy as np
import torch
import cv2
from paddleocr import PPStructure
from paddleocr.ppstructure.table.predict_table import to_excel
from utils.plots import Annotator, colors
FILE = Path(__file__).resolve()
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (non_max_suppression, scale_coords, check_img_size)
from utils.torch_utils import select_device
from flask import Flask, request, send_from_directory

table_engine = PPStructure(show_log=True)
basedir = os.path.dirname(__file__)



def make_output(Path: List):
    for p in Path:
        if not os.path.exists(p):
            os.makedirs(p)
        else:
            continue
    return True

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf8')
        return json.JSONEncoder.default(self, obj)

class TableAnalysis():
    """/home/ubuntu18/gzc/DigitalDocumentAnalysisSystem/yolov5-master/runs/train/exp3/weights/best.pt"""
    def __init__(self, weights='/home/hello/gzc/3.project/yolov5-master/runs/train/exp3/weights/best.pt',
                 data='data/coco.yaml',
                 conf_thres=0.25,
                 iou_thres=0.45,
                 max_det=1000,
                 device='',
                 view_img=True,
                 classes=None,
                 imgsz=(640, 640),
                 agnostic_nms=False,  # class-agnostic NMS
                 line_thickness=1,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 dnn= False):
        super(TableAnalysis, self).__init__()

        self.url = 'http://172.16.16.112:8888/'
        self.tab_num = 0

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
        self.upload_file_path = './static/AnalysisSystem/TableAnalysis/upload_file/' + self.time + '/'
        self.upload_file_to_img_path = './static/AnalysisSystem/TableAnalysis/pdf/pdf_to_img/' + self.time + '/'
        self.layout_img = './static/AnalysisSystem/TableAnalysis/layout/' + self.time + '/'

        self.recog_tab_path = './static/AnalysisSystem/TableAnalysis/tab/' + self.time + '/'
        self.excel_path = './static/AnalysisSystem/TableAnalysis/excel/' + self.time + '/'
        self.detect_path = './static/AnalysisSystem/TableAnalysis/detect/' + self.time + '/'

        make_output([self.upload_file_path, self.upload_file_to_img_path, self.layout_img, self.recog_tab_path,
                     self.detect_path, self.excel_path])

    def CutImage(self, axis, Image, page_num, tab_num):
        """获取文档元素像素和"""
        image = cv2.imread(Image)
        dst_img = image[axis[0][1]: axis[1][1], axis[0][0]: axis[1][0]]
        cv2.imwrite(self.recog_tab_path + '{}_{}.png'.format(page_num, tab_num), dst_img)
        tab_url = self.url + self.recog_tab_path.split('.')[-1] + '{}_{}.png'.format(page_num, tab_num)
        # img_bytes = dst_img.tobytes()
        # img_base64 = base64.b64encode(img_bytes)
        return dst_img, tab_url

    def Mod_Pix(self, dst):
        h, w, c = dst.shape[:]
        h_scale_expand = 300 / h
        w_scale_expand = 500 / h
        if h < 300 and w > 500:
            cv2.resize(dst, (h * h_scale_expand, w))
        elif h > 300 and w < 500:
            cv2.resize(dst, (h, w * w_scale_expand))
        elif h < 300 and w < 500:
            cv2.resize(dst, (h * h_scale_expand, w * w_scale_expand))
        return dst

    def table_analysis(self, pgnum, img_path):
        source = str(img_path)
        name = source.split('/')[-1].split('.')[1]
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        num = 0
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
            DATA = list()

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
                            annotator.box_label(xyxy, label, color=colors(int(cls), True))
                            if label == 'Table': #247 * 381 319 * 507
                                dst_img, tab_base64 = self.CutImage(axis, source, pgnum, self.tab_num)
                                #表格结构识别
                                dst_img = expand(dst_img)
                                tab_res = table_engine(dst_img)
                                excel_path = self.excel_path + "{}_{}.xls".format(pgnum, self.tab_num)
                                excel_name = "{}_{}.xls".format(pgnum, self.tab_num)
                                url_excel = basedir + self.excel_path.split('.')[-1]
                                # url_excel = self.url + self.excel_path.split('.')[-1]

                                
                                for res in tab_res[0]:
                                    if res['type'] == 'table':
                                        html_str = res['res']['html']#html表格
                                        to_excel(html_str, excel_path)#转为单元格
                                        data_tmp['html_str'] = html_str
                                        data_tmp['excel_path'] = url_excel
                                        data_tmp['excel_name'] = excel_name
                                        data_tmp['tab_ori'] = tab_base64
                                        data_tmp['tab_id'] = self.tab_num
                                # else:
                                #     html_str = tab_res[0]['res']
                                #     data_tmp['html_str'] = "None"
                                #     data_tmp['excel_path'] = "None"
                                #     data_tmp['tab_ori'] = tab_base64
                                        DATA.append(data_tmp)
                                self.tab_num += 1

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
            page_res = self.table_analysis(pgnum, IMGPath)
            if len(page_res) != 0:
                data_tmp['page_num'] = str(pgnum)
                data_tmp['page_res'] = page_res
                DATA.append(data_tmp)
        # data = json.dumps({'code': 200, 'msg': 'success', 'PDFFile': PDFPath, 'data': DATA},
        #                     # cls=MyEncoder,
        #                     indent=4,
        #                     ensure_ascii=False)
        # with open('TableAnalysis.json', 'w') as json_file:
        #     json_file.write(data)
        data = {'code': 200, 'msg': 'success', 'PDFFile': PDFPath, 'data': DATA}
        # print(data)
        return data

def Mod_Pix(dst):
    w, h, c = dst.shape[:]
    w_scale_expand = 300 / w
    h_scale_expand = 500 / h
    if w < 300 and h > 500:
        dst = cv2.resize(dst, (int(w * w_scale_expand), h))
    elif w > 300 and h < 500:
        dst = cv2.resize(dst, (w, int(h * h_scale_expand)))
    elif w < 300 and h < 500:
        dst = cv2.resize(dst, (int(w * w_scale_expand), int(h * h_scale_expand)))
    return dst

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

def tabel_rec_test():
    """调试代码"""
    img_path = 'table_test/table0.jpg'
    img = cv2.imread(img_path)

    #type为table可生成html type为figure可生成html
    img = expand(img)
    result = table_engine(img)
    # print('result:', result)
    cv2.imwrite('table.jpg', img)
    for res in result:
        if res['type'] == 'table':
            html_str = res['res']['html']
            print("html:", html_str)
            to_excel(html_str, '2.xls')
        else:
            print('no table')
    # if result[0]['type'] == 'table':
    #     html_str = result[0]['res']['html']
    #     print("html:", html_str)
    #     to_excel(html_str, '2.xls')
    # else:
    #     print('no table')

def test():
    pdfpath = 'pdftest/test_3.pdf'
    # pdfpath = 'pdftest/test_5.pdf'
    TA = TableAnalysis()
    TA.main(pdfpath)

if __name__ == '__main__':
    tabel_rec_test()
    # test()