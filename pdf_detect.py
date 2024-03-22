import base64
import shutil
from typing import List

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
from flask import Flask, request, jsonify

app = Flask(__name__)
basepath = os.path.dirname(__file__)


class CHPDFDetect():
    """
    nohup python -u yolo_detect.py >***.log 2>&1 & 20280
    text title figure table euq
    /home/hello/gzc/3.project/yolov5-master/runs/train/exp3/weights
    /home/ubuntu18/gzc/DigitalDocumentAnalysisSystem/yolov5-master/runs/train/exp3/weights/best.pt
    'D:/iscas/1.layout_analysis/yolov5-master/train/runs/exp3/weights/best.pt'
    D:/iscas/1.layout_analysis/yolov5-master/data/coco.yaml
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
            dnn= False):
        super(CHPDFDetect, self).__init__()


        # Load model
        # self.device = 'cpu'
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

        self.upload_file_path = './static/file/ch_pdf/upload_file/' + self.time + '/'
        self.upload_file_to_img_path = './static/file/ch_pdf/pdf_to_img/' + self.time + '/'
        self.layout_img = './static/file/ch_pdf/layout/' + self.time + '/'
        self.recog_fig_path = './static/file/ch_pdf/recog_fig/' + self.time + '/'
        self.recog_tab_path = './static/file/ch_pdf/recog_tab/' + self.time + '/'
        self.recog_equa_path = './static/file/ch_pdf/recog_equa/' + self.time + '/'
        self.detect_path = './static/file/ch_pdf/detet/' + self.time + '/'

        #html
        self.html = basepath + '/static/file/ch_pdf/templates/' + self.time + '/'
        #压缩包
        self.pdf_and_html = './static/file/ch_pdf/pdf_and_html/' + self.time + '/'
        self.zip = basepath + '/static/file/ch_pdf/zip/'

        # self.make_output(self.upload_file_path)
        # self.make_output(self.upload_file_to_img_path)
        # self.make_output(self.layout_img)
        # self.make_output(self.recog_fig_path)
        # self.make_output(self.recog_tab_path)
        # self.make_output(self.recog_equa_path)
        # self.make_output(self.detect_path)
        # self.make_output(self.html)
        # self.make_output(self.zip)
        self.make_output([self.upload_file_path, self.upload_file_to_img_path, self.layout_img, self.recog_fig_path,
                          self.recog_tab_path, self.recog_equa_path, self.detect_path, self.html, self.pdf_and_html, self.zip])

    def make_output(self, path: List):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
        return True

    def gen_axis(self, axis):
        axis_list = list()
        for i in axis:
            w = abs(i[0]-i[2])
            h = abs(i[1]-i[3])
            coordinates = (i[0], i[1])
            axis_list.append((coordinates, w, h, i[0]))
        return axis_list

    def comp_IoU(self, axis_1, axis_2):
        '''
        计算两个bbox的IoU值
        ((x,y),w,h)
        '''
        rec1 = axis_1
        rec2 = axis_2

        x1 = rec1[0][0]
        y1 = rec1[0][1]
        x2 = x1 + rec1[1]
        y2 = y1 + rec1[2]

        a1 = rec2[0][0]
        b1 = rec2[0][1]
        a2 = a1 + rec2[1]
        b2 = b1 + rec2[2]

        left_column_max = max(x1,a1)
        right_column_min = min(x2,a2)
        up_row_max = max(y1,b1)
        down_row_min = min(y2,b2)

        # 两矩形无相交区域的情况
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        # 两矩形有相交区域
        else:
            s1 = (x2 - x1) * (y2 - y1)
            s2 = (a2 - a1) * (b2 - b1)
            s_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
            IOU = s_cross / (s1 + s2 - s_cross)
            return IOU

    def remove_overlap(self, fig_axis_list, tab_axis_list, text_axis_list):
        '''
        文本和图片去重
        文本和表格去重
        '''
        #--------------------------------------文本和图片去重-----------------------------------
        text_and_fig_overlap = list()
        for idx_i,i in enumerate(fig_axis_list):
            for idx_j,j in enumerate(text_axis_list):
                IOU = self.comp_IoU(i[1],j[1])
                if IOU > 0:
                    text_and_fig_overlap.append(idx_j)

        overlap_bbox = list(sorted(set(text_and_fig_overlap)))[::-1]
        for i in overlap_bbox:
            text_axis_list.pop(i)

        #--------------------------------------文本和图片去重-----------------------------------
        text_and_tab_overlap = list()
        for idx_i,i in enumerate(tab_axis_list):
            for idx_j,j in enumerate(text_axis_list):
                IOU = self.comp_IoU(i[1],j[1])
                if IOU > 0:
                    text_and_tab_overlap.append(idx_j)

        overlap_bbox = list(sorted(set(text_and_tab_overlap)))[::-1]
        for i in overlap_bbox:
            text_axis_list.pop(i)

        return text_axis_list

    def remove_sidebar(self, axis_list):
        '''
        对左侧边栏进行删除
        :param axis_list:
        :return:
        '''
        #---------------------------------------规避文章中的侧边栏---------------------------------------
        axis_list.sort(key=lambda b:b[1][0][0])

        x_axis = list()
        for i in range(len(axis_list)):
            x_axis.append((axis_list[i][1][0][0], i))
        #[(0.8640003204345703, 0), (14.36400032043457, 1), (77.29462051391602, 2), (77.30069732666016, 3), (77.30069732666016, 4), (77.30069732666016, 5), (77.30069732666016, 6), (77.30069732666016, 7), (77.30069732666016, 8), (77.3012924194336, 9), (90.73694801330566, 10), (239.30069732666016, 11), (363.71324157714844, 12), (473.2426300048828, 13), (473.24400329589844, 14), (473.24400329589844, 15), (498.3300018310547, 16), (700.4687347412109, 17), (790.5258178710938, 18)]

        x_sum = 0
        for (x, i) in x_axis:
            x_sum += x
        x_average = x_sum / len(x_axis)

        left_x = list()
        left_x_axis = list()
        right_x = list()
        for x,i in x_axis:
            if x < x_average:
                left_x.append((int(x),i))
                left_x_axis.append(int(x))
            elif x > x_average:
                right_x.append((x,i))

        x_num_dict = {}
        for i in range(len(left_x)):
            x_num_dict[left_x[i][0]] = left_x_axis.count(left_x[i][0])
        #{0: 1, 14: 1, 77: 8, 90: 1, 239: 1}

        delete_axis = list()
        for idx,(k,v) in enumerate(x_num_dict.items()):
            if v > 1:
                break
            elif v == 1 and len(axis_list[idx][0][1]) < 15:
                delete_axis.append(k)

        delete_axis_index = list()
        for axis in delete_axis:
            for (x,i) in left_x:
                if x == axis:
                    delete_axis_index.append(i)

        delete_axis_index = delete_axis_index[::-1]
        for i in delete_axis_index:
            axis_list.pop(i)
        #---------------------------------------规避特殊文本---------------------------------------

        text_index = list()
        for i in range(len(axis_list)):
            pattern1 = '图像: 设备RGB|图片:DeviceRGB'
            if re.search(pattern1,axis_list[i][0][1]) is not None:
                text_index.append(i)

        text_index = text_index[::-1]
        for i in text_index:
            axis_list.pop(i)

        return axis_list

    def CutImage(self, img_path, fig_bbox_list, tab_bbox_list, fig_save_path, tab_save_path):
        '''
        将bbox图片中对应的位置裁剪下来
        '''
        fig_list = list()
        fig_bbox = list()

        tab_list = list()
        tab_bbox = list()

        fig_axis_list = list()
        tab_axis_list = list()

        filename = img_path.split('/')[-1].split('.')[0]
        image = cv2.imread(img_path)
        if len(fig_bbox_list) != 0:
            for idx, i in enumerate(fig_bbox_list):
                label = i[1]
                axis = i[0]
                fig_path = fig_save_path + '/' + filename + '_fig_{}.jpg'.format(idx)
                dst_img = image[axis[0][1]:axis[1][1], axis[0][0]:axis[1][0]]
                cv2.imwrite(fig_path, dst_img)
                fig_base64 = base64.b64encode(open(fig_path, 'rb').read())
                fig_base64 = fig_base64.decode()
                fig_list.append((fig_base64, label))

                coordinates = (axis[0][0], axis[0][1])
                w = abs(axis[0][0] - axis[1][0])
                h = abs(axis[0][1] - axis[1][1])
                fig_bbox.append((coordinates, w, h))
            fig_axis_list = list(zip(fig_list, fig_bbox))

        if len(tab_bbox_list) != 0:
            for idx, i in enumerate(tab_bbox_list):
                label = i[1]
                axis = i[0]
                tab_path = tab_save_path + '/' + filename + '_tab_{}.jpg'.format(idx)
                dst_img = image[axis[0][1]:axis[1][1], axis[0][0]:axis[1][0]]
                cv2.imwrite(tab_path, dst_img)
                tab_base64 = base64.b64encode(open(tab_path, 'rb').read())
                tab_base64 = tab_base64.decode()
                tab_list.append((tab_base64, label))

                coordinates = (axis[0][0], axis[0][1])
                w = abs(axis[0][0] - axis[1][0])
                h = abs(axis[0][1] - axis[1][1])
                tab_bbox.append((coordinates, w, h))
            tab_axis_list = list(zip(tab_list, tab_bbox))

        return fig_axis_list, tab_axis_list

    def YOLODetect(self, img_path):

        fig_bbox_list = list()
        tab_bbox_list = list()
        equa_bbox_list = list()

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
                            if label == "Figure" or label == 'Equation':
                                fig_bbox_list.append((axis, label))
                            elif label == "Table":
                                tab_bbox_list.append((axis, label))
                            elif label == "Equation":
                                equa_bbox_list.append((axis, label))

                im0 = annotator.result()
                cv2.imwrite(self.detect_path + '/' + name + '.png', im0)

        #裁剪文档目标 转为base64
        fig_axis_list, tab_axis_list = self.CutImage(source, fig_bbox_list, tab_bbox_list, self.recog_fig_path, self.recog_tab_path)
        return fig_axis_list, tab_axis_list


    def single_PDF_detection(self, img_path):
        '''
        对单篇pdf布局检测 返回需求布局结果
        '''
        # from engine.single_pdf_detection import main_detection_evaluation
        fig_bbox_list = list()
        tab_bbox_list = list()

        label_list = main_detection_evaluation(img_path)
        for i in label_list:
            if i[1] == 'List':
                fig_bbox_list.append((i[0], 'Figure'))
            if i[1] == 'Figure':
                fig_bbox_list.append((i[0], i[1]))
            if i[1] == 'Table':
                tab_bbox_list.append((i[0], i[1]))

        fig_axis_list, tab_axis_list = self.CutImage(img_path, fig_bbox_list,
                                                    tab_bbox_list, self.recog_fig_path,
                                                    self.recog_tab_path)

        return fig_axis_list, tab_axis_list

    def gen_absolute_positioning_css(self,text_axis_list):
        '''
        :return:
        '''
        divstring = ''
        for i in text_axis_list:
            text = i[0][1]
            text_type = i[0][-1]
            axis = i[1]
            if text_type == 'Text':
                cssstr = 'left:{0}px;' \
                         'top:{1}px;' \
                         'width:{2}px;' \
                         'font-size:10px;' \
                         'font-family:SimSun;' \
                         'background: white; ' \
                         'margin: auto;' \
                         'border: 1px solid black;' \
                         'position: absolute;'.format(
                    axis[0][0], axis[0][1], axis[1]
                )
                divstring += '<div style="{0}">{1}</div>'.format(cssstr, text)

        return divstring

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
                #单栏中的文本和标题在CSS样式中的宽度设置为占父类像素100%，图像、表格、列表在CSS样式中宽高值为原图片裁切下来的像素大小，并设置一个margin-bottom为10.
                if bbox_list[1] == 'main':#width: {0}%占父类像素100%
                    if item[0][-1] == 'Title':
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;' \
                                 'float: left; '.format(100, title_font_size)#float(item[1][1]/pdf_w) * 100
                        divstring += '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1], cssstr, item[0][1])
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
                    #左栏中的文本和标题在CSS样式中宽度为[(Wbbox/Wl)*100],Wbbox是文本框的宽度，图片、表格、列表CSS样式中的参数同单栏一样。
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
                                 'margin-bottom: {2}px'.format((float(item[1][1]) / left_w_max) * 100, text_font_size, margin)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

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
                                    item[1][1],item[1][2],margin
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
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr,item[0][1])

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
                divstring = '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1],cssstr,divstring)
            elif bbox_list[1] == 'right' and right_w_max != 0:
                cssstr = 'width: {0}%;' \
                         'float: right;' \
                         'padding-top:{1}px;' \
                         'box-sizing: border-box;'.format((right_w_max / (pdf_w - main_padding_list[0][0][0]*2))*100
                                                        , 10)
                divstring = '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1],cssstr,divstring)

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

    def gen_css(self, text_axis_list, html_path, pdf_w, pdf_h, figure_axis_list=None, table_axis_list=None):
        '''
        :param figure_axis_list:
        :param table_axis_list:
        :param text_axis_list:
        :param html_path:
        :param pdf_w:
        :param pdf_h:
        :return:
        '''
        text_axis_list.sort(key=lambda b:(b[1][0][0],b[1][0][1]))
        if len(figure_axis_list) != 0 and len(table_axis_list) != 0:
            bbox_list = figure_axis_list + table_axis_list + text_axis_list
        if len(figure_axis_list) != 0 and len(table_axis_list) == 0:
            bbox_list = figure_axis_list + text_axis_list
        if len(figure_axis_list) == 0 and len(table_axis_list) != 0:
            bbox_list = table_axis_list + text_axis_list
        if len(figure_axis_list) == 0 and len(table_axis_list) == 0:
            bbox_list = text_axis_list

        bbox_blocks = self.three_bbox_sort(bbox_list, pdf_w)
        divstring = self.css_div(bbox_blocks, pdf_w)

        # HTMLstr = "<head></head><body>{0}</body>".format(divstring)
        # HTMLstr = "<html><head></head><body>{0}</body></html>".format(divstring)

        # with open(html_path, 'w',encoding='utf8') as f:
        #     f.write(HTMLstr)

        # pdfpath = self.html_to_pdf(html_path, pdf_file_output)

        # return html_path, pdfpath
        return divstring

    def layout_parser(self, img_path, text_axis_list, html_path, pdf_w, pdf_h):
        '''
        布局分析
        '''
        #获取图片表格
        # fig_axis_list, tab_axis_list = self.single_PDF_detection(img_path)
        fig_axis_list, tab_axis_list = self.YOLODetect(img_path)
        self.YOLODetect(img_path)

        #去重
        text_axis_list = self.remove_overlap(fig_axis_list, tab_axis_list, text_axis_list)

        #生成样式
        DIVstr = self.gen_css(text_axis_list, html_path, pdf_w, pdf_h, fig_axis_list, tab_axis_list)
        # DIVstr = self.gen_absolute_positioning_css(text_axis_list)

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
        #-----------------------------------获取文本----------------------------------------
            pdf_scale_h = float(page.CropBox.height) * 1.5
            pdf_scale_w = float(page.CropBox.width) * 1.5

            text_blocks = page.get_text('blocks')
            text_blocks.sort(key=lambda b: (b[0],b[1]))

            text_bboxes = list()
            text_list = []
            for b in text_blocks:
                w = abs(b[0] - b[2])
                h = abs(b[1] - b[3])
                coordinates = (b[0] * 1.5, b[1] * 1.5)
                text_bboxes.append((coordinates, w * 1.5, h * 1.5))
                # print(b[4])
                text_list.append((b[4], b[4], 'Text'))
            text_axis_list = list(zip(text_list, text_bboxes))
            text_axis_list = self.remove_sidebar(text_axis_list)
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
            divstr = self.layout_parser(img_path, text_axis_list, html_path, pdf_scale_w, pdf_scale_h)
            DIVstr += ''.join(divstr)
        HTMLstr = "<html><head></head><body>{0}</body></html>".format(DIVstr)
        with open(html_path, 'w', encoding='utf8') as f:
            f.write(HTMLstr)


        #
        if os.path.isfile(PdfPath):
            shutil.copy(PdfPath, self.pdf_and_html)
        if os.path.isfile(html_path):
            shutil.copy(html_path, self.pdf_and_html)
        shutil.make_archive(self.zip + self.time, 'zip', self.pdf_and_html)

        data = {
                "FileName": pdf_name,
                "HTMLPath": html_path,
                "ZIPPath": self.zip + self.time + '.zip'
                }
        # print(data)
        return data
        # return {'code': 200, 'msg': 'succsee', 'data': HTMLstr}

def main():
    file = 'pdftest/test_6.pdf'
    filename = file.split('/')[-1].split('.')[0]
    filetype = file.split('/')[-1].split('.')[-1]

    ch = CHPDFDetect()
    pdf_path = ch.upload_file_path#pdf保存地址
    pdf_abspath = pdf_path + filename + '.' + filetype
    pdf_img = ch.upload_file_to_img_path
    # file.save(pdf_abspath)
    num = random.randint(0, 100000000)
    html = ch.html
    html_path = html + str(num) + '.html'

    data = ch.pyMuPDF_fitz(file, pdf_img, html_path)

#http://39.104.86.105:8020/ch_label
#http://172.16.16.112:50088/ch_label
#http://172.16.20.42:50088/ch_label

@app.route('/pdf2html', methods=['GET', 'POST'])
def pdf2html():

    file = request.files.get('pdf2html')
    file_name = file.filename.split('.')[0]
    file_type = file.filename.split('.')[1]

    CH = CHPDFDetect()
    pdf_path = CH.upload_file_path#pdf保存地址
    pdf_abspath = pdf_path + file_name + '.' + file_type
    pdf_img = CH.upload_file_to_img_path
    file.save(pdf_abspath)

    num = random.randint(0, 100000000)
    html = CH.html
    html_path = html + str(num) + '.html'

    # data = CH.pyMuPDF_fitz(pdf_abspath,pdf_img,html_path)
    #
    # return data

    try:
        data = CH.pyMuPDF_fitz(pdf_abspath, pdf_img, html_path)
        return jsonify(data)
    except Exception as e:
        print(e)
        data = {'code': 500, 'msg': 'error', 'data': 'PDF parsing failed'}
        return jsonify(data)

@app.route('/ch_label',methods=['GET', 'POST'])
def CH_label():

    file = request.files.get('ChPDFFile')
    file_name = file.filename.split('.')[0]
    file_type = file.filename.split('.')[1]

    CH = CHPDFDetect()
    pdf_path = CH.upload_file_path#pdf保存地址
    pdf_abspath = pdf_path + file_name + '.' + file_type
    pdf_img = CH.upload_file_to_img_path
    file.save(pdf_abspath)

    num = random.randint(0, 100000000)
    html = CH.html
    html_path = html + str(num) + '.html'

    # data = CH.pyMuPDF_fitz(pdf_abspath,pdf_img,html_path)
    #
    # return data

    try:
        data = CH.pyMuPDF_fitz(pdf_abspath, pdf_img, html_path)
    except Exception as e:
        print(e)
        data = {'code': 500, 'msg': 'error', 'data': 'PDF parsing failed'}
    return data

if __name__ == '__main__':
    main()
    # app.run('0.0.0.0', port=10010)

#conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cpuonly -c pytorch