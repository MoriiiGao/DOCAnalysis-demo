import base64
import fitz
import urllib.request
import urllib.parse
import os
import random
import re
import string
import time
import urllib
from pathlib import Path
import torch
import cv2
from paddleocr import PaddleOCR
from utils.plots import Annotator, colors
FILE = Path(__file__).resolve()
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from flask import Flask, request
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

app = Flask(__name__)

class CHPDFDetect():
    """
    nohup python -u yolo_detect.py >***.log 2>&1 & 20280
    text title figure table euq
    /home/hello/gzc/3.project/yolov5-master/runs/train/exp3/weights
    'D:/iscas/1.layout_analysis/yolov5-master/train/runs/exp3/weights/best.pt'
    D:/iscas/1.layout_analysis/yolov5-master/data/coco.yaml
    """
    def __init__(self, weights='runs/train/exp3/weights/best.pt',  # model.pt path(s)
            data='D:/iscas/1.layout_analysis/yolov5-master/data/coco.yaml',  # dataset.yaml path
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
            dnn= False):
        super(CHPDFDetect, self).__init__()

        # Load model
        self.device = 'cpu'
        # self.device = select_device(device)
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

        self.upload_file_path = './static/file/ch_pdf/upload_file/' + self.time + '/'
        self.upload_file_to_img_path = './static/file/ch_pdf/pdf_to_img/' + self.time + '/'
        self.layout_img = './static/file/ch_pdf/layout/' + self.time + '/'
        self.recog_fig_path = './static/file/ch_pdf/recog_fig/' + self.time + '/'
        self.recog_tab_path = './static/file/ch_pdf/recog_tab/' + self.time + '/'
        self.recog_equa_path = './static/file/ch_pdf/recog_equa/' + self.time + '/'
        self.recog_text_path = './static/file/ch_pdf/recog_text/' + self.time + '/'
        self.recog_title_path = './static/file/ch_pdf/recog_title/' + self.time + '/'

        self.detect_path = './static/file/ch_pdf/detet/' + self.time + '/'
        #html
        self.html = './static/file/ch_pdf/templates/' + self.time + '/'

        self.make_output(self.upload_file_path)
        self.make_output(self.upload_file_to_img_path)
        self.make_output(self.layout_img)
        self.make_output(self.recog_text_path)
        self.make_output(self.recog_title_path)
        # self.make_output(self.recog_fig_path)
        # self.make_output(self.recog_tab_path)
        self.make_output(self.recog_equa_path)
        self.make_output(self.detect_path)
        self.make_output(self.html)

    def make_output(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return True

    def np_to_base64(self, dst_img):
        """numpy转base64"""
        data = cv2.imencode('.jpg', dst_img)[1]
        img_bytes = data.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf8')
        return img_base64

    def border_extension(self, img):
        """
        问题：paddleocr无法精准识别小图和长图
        方法：对图片进行预处理 增加图片的大小 给图片加一定大小的框
        对裁剪出来的小图和长图 在边界填充一定数目的像素
        """
        h, w, c = img.shape[:]
        border = [0, 0]
        transform_size = 320 #图片增加边框到320大小
        if w < transform_size or h < transform_size:
            if h < transform_size:
                border[0] = (transform_size - h) / 2.0
            if w < transform_size:
                border[0] = (transform_size - 2) / 2.0

            #top, buttom, left, right对应边界的像素数目
            img = cv2.copyMakeBorder(img, int(border[0]), int(border[0]), int(border[1]), int(border[1]),
                                         cv2.BORDER_CONSTANT,
                                         value=[215, 215, 215])
        return img

    def get_main_or_left_axis(self, bbox_blocks):
        '''
        获取整页布局第一个main或left类型div的x和y坐标
        '''
        main_padding_list = list()  # 存放第一个main或left类型div的x和y坐标
        for i, bbox_list in enumerate(bbox_blocks):
            for j, item in enumerate(bbox_list[0]):
                if (bbox_list[1] == 'main' or bbox_list[1] == 'left') and len(main_padding_list) == 0:
                    main_padding_list.append(((item[1][0][0], item[1][0][1]), item[1][1], item[1][2]))
        return main_padding_list

    def OCR(self, dst_img):

        img = self.border_extension(dst_img)
        result = ocr.ocr(img, cls=True)
        text = ""
        for line in result:
            if len(line) != 0:
                for i in line:
                    txt = i[1][0]
                    text += txt
            else:
                text = " "
        return text

    def CutImage(self, img, object_list):

        fig_axis_list = list()
        table_axis_list = list()
        text_axis_list = list()
        title_axis_list = list()

        img = cv2.imread(img)
        for idx, i in enumerate(object_list):
            axis = i[0]
            label = i[1]
            if label == 'Figure' or label == "Equation":
                dst_img = img[axis[0][1]:axis[1][1], axis[0][0]:axis[1][0]]
                img_base64 = self.np_to_base64(dst_img)
                fig_axis_list.append(((img_base64, "Figure"), ((axis[0][0], axis[0][1]), abs(axis[1][0]-axis[0][0]), abs(axis[0][1]-axis[1][1]) )))

            elif label == "Table":
                dst_tab = img[axis[0][1]:axis[1][1], axis[0][0]:axis[1][0]]
                tab_base64 = self.np_to_base64(dst_tab)
                table_axis_list.append(((tab_base64, label), ((axis[0][0], axis[0][1]), abs(axis[1][0]-axis[0][0]), abs(axis[0][1]-axis[1][1]))))

            elif label == "Text" or label == 'Reference':
                dst_text = img[axis[0][1]:axis[1][1], axis[0][0]:axis[1][0]]
                text_path = self.recog_text_path + '/' + '{}.jpg'.format(idx)
                cv2.imwrite(text_path, dst_text)
                text = self.OCR(dst_text)
                text_axis_list.append(((text, text, 'Text'), ((axis[0][0], axis[0][1]), abs(axis[1][0]-axis[0][0]), abs(axis[0][1]-axis[1][1]))))

            elif label == "Title":
                dst_title = img[axis[0][1]:axis[1][1], axis[0][0]:axis[1][0]]
                text_path = self.recog_title_path + '/' + '{}.jpg'.format(idx)
                cv2.imwrite(text_path, dst_title)
                title = self.OCR(dst_title)
                title_axis_list.append(((title, title, label), ((axis[0][0], axis[0][1]), abs(axis[1][0]-axis[0][0]), abs(axis[0][1]-axis[1][1]))))

        return fig_axis_list, table_axis_list, text_axis_list, title_axis_list

    def YOLODetect(self, img_path):

        source = str(img_path)
        name = source.split('/')[-1].split('.')[1]
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        result = list()
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

            #Process predictions
            for i, det in enumerate(pred): #per image
                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        if self.view_img:  # Add bbox to image
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            label = self.names[int(cls)]
                            axis = ((x1, y1), (x2, y2))
                            score = f'{conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(int(cls), True))
                            result.append((axis, label))

                im0 = annotator.result()
                cv2.imwrite(self.detect_path + '/' + name + '.png', im0)

        return result

    def three_bbox_sort(self, bbox_list, pdf_w):
        '''
        将bbox分为三种情况：单独一栏的[横跨两列]、在左列、在右列
        :param bbox_list:
        :return:
        '''
        left_blocks = list()
        left_x = list()
        right_blocks =list()
        right_x = list()
        for i in range(len(bbox_list)):
            if bbox_list[i][1][0][0] < pdf_w/2:
                left_x.append(bbox_list[i][1][0][0])
            elif bbox_list[i][1][0][0] > pdf_w/2:
                right_x.append(bbox_list[i][1][0][0])

        for i in range(len(bbox_list)):
            if bbox_list[i][1][0][0] < pdf_w/2:
                bbox_item = bbox_list[i]
                bbox_item_list = list(bbox_item)
                bbox_w_h = bbox_item_list[1]
                bbox_w_h_list = list(bbox_w_h)
                left_corner_axis = bbox_w_h_list[0]
                left_corner_axis_list = list(left_corner_axis)

                left_corner_axis_list[0] = min(left_x)

                left_corner_axis = tuple(left_corner_axis_list)
                bbox_w_h_list[0] = tuple(left_corner_axis)
                bbox_w_h = tuple(bbox_w_h_list)
                bbox_item_list[1] = bbox_w_h
                bbox_list[i] = tuple(bbox_item_list)
                left_blocks.append(bbox_list[i])

            elif bbox_list[i][1][0][0] > pdf_w/2:
                bbox_item = bbox_list[i]
                bbox_item_list = list(bbox_item)
                bbox_w_h = bbox_item_list[1]
                bbox_w_h_list = list(bbox_w_h)
                right_corner_axis = bbox_w_h_list[0]
                right_corner_axis_list = list(right_corner_axis)

                right_corner_axis_list[0] = min(right_x)

                right_corner_axis = tuple(right_corner_axis_list)
                bbox_w_h_list[0] = right_corner_axis
                bbox_w_h = tuple(bbox_w_h_list)
                bbox_item_list[1] = bbox_w_h
                bbox_list[i] = tuple(bbox_item_list)
                right_blocks.append(bbox_list[i])
        # ----------------------------------从左侧栏中筛选除中间栏--------------------------------------
        single_line_blocks = list()
        del_left_blocks = list()
        for i in range(len(left_blocks)):
            width = left_blocks[i][1][1]
            x = left_blocks[i][1][0][0]
            if x + width > (pdf_w / 2):#
            # if x + width > (pdf_w / 2) + 40:#
                single_line_blocks.append(left_blocks[i])
                del_left_blocks.append(i)

        del_left_blocks = del_left_blocks[::-1]
        for i in del_left_blocks:
            left_blocks.pop(i)

        #按y轴对每个列表中的bbox进行排序
        single_line_blocks.sort(key=lambda b: b[1][0][1])
        left_blocks.sort(key=lambda b: b[1][0][1])
        right_blocks.sort(key=lambda b: b[1][0][1])

        all_bbox_list = list()
        for single_block in single_line_blocks:
            all_bbox_list.append((single_block, 'main'))
        for left_block in left_blocks:
            all_bbox_list.append((left_block, 'left'))
        for right_block in right_blocks:
            all_bbox_list.append((right_block, 'right'))

        all_bbox_list.sort(key=lambda b: b[0][1][0][1])

        new_bbox_list = list()
        main_list = list()
        left_list = list()
        right_list = list()
        for bbox in all_bbox_list:
            if bbox[1] == 'left':
                left_list.append(bbox[0])
                if len(main_list) != 0:
                    new_bbox_list.append((main_list, 'main'))
                    main_list = list()

            elif bbox[1] == 'right':
                right_list.append(bbox[0])
                if len(main_list) != 0:
                    new_bbox_list.append((main_list, 'main'))
                    main_list = list()

            elif bbox[1] == 'main':#标识符为main的时候 将left和right分别存进一个列表 然后更新一个新列表
                main_list.append(bbox[0])
                if len(left_list) != 0:
                    new_bbox_list.append((left_list, 'left'))
                    left_list = list()
                if len(right_list) != 0:
                    new_bbox_list.append((right_list, 'right'))
                    right_list = list()

        #将保存的列表保存
        if len(main_list) != 0:
            new_bbox_list.append((main_list, 'main'))
        if len(left_list) != 0:
            new_bbox_list.append((left_list, 'left'))
        if len(right_list) != 0:
            new_bbox_list.append((right_list, 'right'))

        return new_bbox_list

    def css_div(self, bbox_blocks, pdf_w):
        '''
        css样式模板
        :param bbox_list:
        :param pdf_w:
        :return:
        '''
        text_font_size = 14
        title_font_size = 20
        divstring = ''
        div_string = list()

        Max_left_block_w = list()
        Max_right_block_w = list()
        for bbox_list in bbox_blocks:
            if bbox_list[1] == 'left':
                for item in bbox_list[0]:
                    Max_left_block_w.append(item[1][1])

            if bbox_list[1] == 'right':
                for item in bbox_list[0]:
                    Max_right_block_w.append(item[1][1])

        if len(Max_left_block_w) != 0:
            left_w_max = max(Max_left_block_w)
        else:
            left_w_max = 0

        if len(Max_right_block_w) != 0:
            right_w_max = max(Max_right_block_w)
        else:
            right_w_max = 0

        # try:
        #     left_w_max = max(Max_left_block_w)#左边block中 所有宽度的最大值
        # except Exception as e:
        #     pass
        #
        # try:
        #     right_w_max = max(Max_right_block_w)
        # except Exception as e:
        #     pass

        main_padding_list = self.get_main_or_left_axis(bbox_blocks)
        # 进入大列表for循环

        for i, bbox_list in enumerate(bbox_blocks):
            divstring = ''
            #循环每一个小列表
            for j, item in enumerate(bbox_list[0]):
                if j != len(bbox_list[0])-1:
                    # margin = abs(bbox_list[0][j+1][1][0][1] - (item[1][0][1] + item[1][2]))
                    margin = 10
                else:
                    margin = 0

                if bbox_list[1] == 'main':#
                    if item[0][-1] == 'Title':
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;' \
                                 'float: left; '.format(100, title_font_size)#float(item[1][1]/pdf_w) * 100
                        divstring += '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1],cssstr,item[0][1])
                    elif item[0][-1] == 'Text':
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'float: left; '.format(100, text_font_size)
                        divstring += '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1], cssstr, item[0][1])

                    elif item[0][-1] == 'Figure':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'float: left; ' \
                                 'margin-bottom: {2}px'.format(
                                    item[1][1], item[1][2], margin
                            )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    elif item[0][-1] == 'Table':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'float: left; ' \
                                 'margin-bottom: {2}px'.format(
                                    item[1][1], item[1][2], margin
                            )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    # elif item[0][-1] == 'Figure':
                    #     cssstr = 'width:{0}px;' \
                    #              'height:{1}px; ' \
                    #              'float: left; ' \
                    #              'margin-bottom: {2}px;'\
                    #              'margin-left: {3}px'.format(
                    #                 item[1][1],item[1][2],margin,item[1][3] - item[1][0][0]
                    #         )
                    #     divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    # elif item[0][-1] == 'Table':
                    #     cssstr = 'width:{0}px;' \
                    #              'height:{1}px; ' \
                    #              'float: left; ' \
                    #              'margin-bottom: {2}px;' \
                    #              'margin-left: {3}px'.format(
                    #         item[1][1], item[1][2], margin, item[1][3] - item[1][0][0]
                    #     )
                    #     divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                elif bbox_list[1] == 'left' and left_w_max != 0:
                    if len(main_padding_list) == 0:
                        main_padding_list.append(((item[1][0][0], item[1][0][1]), item[1][1], item[1][2]))
                    if item[0][-1] == 'Title' and j == 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;'.format((float(item[1][1]) / left_w_max) * 100, title_font_size)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Text' and j == 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;'.format((float(item[1][1]) / left_w_max) * 100, text_font_size)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Title' and j != 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;' \
                                 'margin-bottom: {2}px'.format((float(item[1][1]) / left_w_max) * 100, title_font_size, margin)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Text' and j != 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'margin-bottom: {2}px'.format((float(item[1][1]) / left_w_max) * 100, text_font_size,margin)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Figure':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'margin-bottom: {2}px;'.format(item[1][1], item[1][2], margin)
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    elif item[0][-1] == 'Table':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'margin-bottom: {2}px;'.format(
                                    item[1][1], item[1][2], margin
                            )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                elif bbox_list[1] == 'right' and right_w_max != 0:
                    if item[0][-1] == 'Title' and j == 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;'.format((float(item[1][1]) / right_w_max) * 100, title_font_size)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr,item[0][1])

                    elif item[0][-1] == 'Text' and j == 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;'.format((float(item[1][1]) / right_w_max) * 100, text_font_size)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr,item[0][1])

                    elif item[0][-1] == 'Title' and j != 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;' \
                                 'margin-bottom: {2}px'.format((float(item[1][1]) / right_w_max) * 100, title_font_size,
                                                               margin)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Text' and j != 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'margin-bottom: {2}px'.format((float(item[1][1]) / right_w_max) * 100, text_font_size,
                                                               margin)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr,item[0][1])

                    elif item[0][-1] == 'Figure':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'margin-bottom: {2}px;'.format(
                            item[1][1], item[1][2], margin
                        )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    elif item[0][-1] == 'Table':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'margin-bottom: {2}px;'.format(
                            item[1][1], item[1][2], margin
                        )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)


            #padding-top计算main最后一个div与left或right第一个div的差值
            #main最后一个div的y坐标:main_padding_list[-1][0][1]
            #main最后一个div的高度h:main_padding_list[-1][2]
            #left第一个div的y坐标:bbox_list[0][0][1][2]
            #退出小模块for循环 main_padding_list[-1][1]
            #left and right padding_top:abs(bbox_list[0][0][1][2] - (main_padding_list[-1][0][1] + main_padding_list[-1][2]))
            if bbox_list[1] == 'left' and left_w_max != 0:
                cssstr = 'width: {0}%;' \
                         'float: left;' \
                         'padding-top:{1}px;' \
                         'box-sizing: border-box;'.format((left_w_max / (pdf_w - main_padding_list[0][0][0]*2))*100
                                                          , 10)
                divstring = '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1], cssstr, divstring)
            elif bbox_list[1] == 'right' and right_w_max != 0:
                cssstr = 'width: {0}%;' \
                         'float: right;' \
                         'padding-top:{1}px;' \
                         'box-sizing: border-box;'.format((right_w_max / (pdf_w - main_padding_list[0][0][0]*2))*100, 10)
                divstring = '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1], cssstr, divstring)

            div_string.append(divstring)

        DIVsrting = ' '.join(div_string)
        #最外层加一个padding
        cssstr = 'width: {0}px;' \
                 'height: 100%;' \
                 'box-sizing: border-box;' \
                 'padding-left: {1}px;' \
                 'padding-top:{2}px;'\
                 'padding-right:{3}px'.format(pdf_w,
                main_padding_list[0][0][0],main_padding_list[0][0][1],main_padding_list[0][0][0])
        DIV_string = '<div  style="{0}">{1}</div><hr style="width:100%"/>'.format(cssstr, DIVsrting)

        return DIV_string


    def gen_css(self, text_axis_list,  figure_axis_list, table_axis_list, title_axis_list, pdf_w):
        '''
        :param figure_axis_list:
        :param table_axis_list:
        :param text_axis_list:
        :param html_path:
        :param pdf_w:
        :param pdf_h:
        :return:
        '''
        text_axis_list.sort(key=lambda b: (b[1][0][0], b[1][0][1]))
        if len(figure_axis_list) != 0 and len(table_axis_list) != 0 and len(title_axis_list) != 0:
            bbox_list = figure_axis_list + table_axis_list + text_axis_list + title_axis_list
        elif len(figure_axis_list) !=0 and len(title_axis_list) != 0 and len(table_axis_list) == 0:
            bbox_list = figure_axis_list + title_axis_list + text_axis_list
        elif len(figure_axis_list) !=0 and len(table_axis_list) != 0 and len(title_axis_list) == 0:
            bbox_list = figure_axis_list + table_axis_list + text_axis_list
        elif len(figure_axis_list) == 0 and len(table_axis_list) != 0 and len(title_axis_list) != 0:
            bbox_list = table_axis_list + title_axis_list + text_axis_list
        elif len(figure_axis_list) != 0 and len(table_axis_list) == 0 and len(title_axis_list) == 0:
            bbox_list = figure_axis_list + text_axis_list
        elif len(figure_axis_list) == 0 and len(table_axis_list) != 0 and len(title_axis_list) == 0:
            bbox_list = table_axis_list + text_axis_list
        elif len(figure_axis_list) == 0 and len(table_axis_list) == 0 and len(title_axis_list) != 0:
            bbox_list = title_axis_list + text_axis_list
        elif len(figure_axis_list) == 0 and len(table_axis_list) == 0 and len(title_axis_list) == 0:
            bbox_list = text_axis_list

        bbox_blocks = self.three_bbox_sort(bbox_list, pdf_w)
        divstring = self.css_div(bbox_blocks, pdf_w)

        return divstring

    def layout_parser(self, img_path, html_path, pdf_w, pdf_h):
        '''
        布局分析
        '''
        all_file_object = self.YOLODetect(img_path)

        fig_axis_list, table_axis_list, text_axis_list, title_axis_list = self.CutImage(img_path, all_file_object)

        DIVstr = self.gen_css(text_axis_list, fig_axis_list, table_axis_list, title_axis_list, pdf_w)
        return DIVstr


    def pyMuPDF_fitz(self, PdfPath, ImagePath, html_path):
        '''
        :param PdfPath:
        :param ImagePath:
        :param html_path:
        :return:
        '''

        pdf_name = PdfPath.split('/')[-1].split('.')[0]
        pdfDoc = fitz.open(PdfPath)

        DIVstr = ''
        for pg in range(pdfDoc.pageCount):
            page = pdfDoc[pg]

            pdf_scale_h = float(page.CropBox.height) * 1.5
            pdf_scale_w = float(page.CropBox.width) * 1.5

        #-----------------------------------pdf转图片---------------------------------------
            rotate = int(0)
            zoom_x = 1.0
            zoom_y = 1.0
            mat = fitz.Matrix(zoom_x,zoom_y).preRotate(rotate)
            pix = page.getPixmap(matrix=mat)
            img_h = pix.height
            img_w = pix.width

            zoom_scale_x = float(pdf_scale_w) / float(img_w)
            zoom_scale_y = float(pdf_scale_h) / float(img_h)
            mat_scale = fitz.Matrix(zoom_scale_x,zoom_scale_y).prerotate(rotate)
            pix_scale = page.getPixmap(matrix=mat_scale,alpha=False)
            if not os.path.exists(ImagePath):
                os.makedirs(ImagePath)

            img_path = ImagePath + '/' + pdf_name + '_' + 'images_%s.png' % pg
            pix_scale.writePNG(img_path) #将图片写入指定的文件夹内

        #-----------------------------------版面分析----------------------------------------
            divstr = self.layout_parser(img_path, html_path, pdf_scale_w, pdf_scale_h)

            DIVstr += ''.join(divstr)
        HTMLstr = "<html><head></head><body>{0}</body></html>".format(DIVstr)
        with open(html_path, 'w', encoding='utf8') as f:
            f.write(HTMLstr)

        return {'code': 200, 'msg': 'succsee', 'data': html_path}

def main():
    #ch_1.pdf
    file = 'pdftest/ch_1.pdf'
    ch = CHPDFDetect()
    pdf_img = ch.upload_file_to_img_path
    num = random.randint(0, 100000000)
    html = ch.html
    html_path = html + str(num) + '.html'

    data = ch.pyMuPDF_fitz(file, pdf_img, html_path)
    print(data)
if __name__ == '__main__':
    main()