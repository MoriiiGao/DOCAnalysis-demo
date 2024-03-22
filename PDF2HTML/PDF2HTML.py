import base64
import fitz
import os
import random
import time
from pathlib import Path
import torch
import cv2
from utils.plots import Annotator, colors
FILE = Path(__file__).resolve()
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from HTMLTemplate import gen_css
from flask import Flask, request
from OCRProcess import OCR
# from paddleocr import paddleocr
#
# ocr = paddleocr(use_angle_cls=True, lang="ch")

app = Flask(__name__)

class CHPDFDetect():
    """
    nohup python -u yolo_detect.py >***.log 2>&1 & 20280
    text title figure table euq
    /home/hello/gzc/3.project/yolov5-master/runs/train/exp3/weights
    'D:/iscas/1.layout_analysis/yolov5-master/train/runs/exp3/weights/best.pt'
    D:/iscas/1.layout_analysis/yolov5-master/data/coco.yaml
    """
    def __init__(self, weights='/home/hello/gzc/3.project/yolov5-master/runs/train/exp3/weights/best.pt',  # model.pt path(s)
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
                text = OCR(dst_text)
                text_axis_list.append(((text, text, 'Text'), ((axis[0][0], axis[0][1]), abs(axis[1][0]-axis[0][0]), abs(axis[0][1]-axis[1][1]))))

            elif label == "Title":
                dst_title = img[axis[0][1]:axis[1][1], axis[0][0]:axis[1][0]]
                text_path = self.recog_title_path + '/' + '{}.jpg'.format(idx)
                cv2.imwrite(text_path, dst_title)
                title = OCR(dst_title)
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

    def layout_parser(self, img_path, html_path, pdf_w, pdf_h):
        '''
        布局分析
        '''
        all_file_object = self.YOLODetect(img_path)

        fig_axis_list, table_axis_list, text_axis_list, title_axis_list = self.CutImage(img_path, all_file_object)

        DIVstr = gen_css(text_axis_list, fig_axis_list, table_axis_list, title_axis_list, pdf_w)
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

        return {'code': 200, 'msg': 'succsee', 'data': HTMLstr}

def main():
    #ch_1.pdf
    file = '/home/hello/gzc/3.project/yolov5-master/pdftest/test_3.pdf'
    ch = CHPDFDetect()
    pdf_img = ch.upload_file_to_img_path
    num = random.randint(0, 100000000)
    html = ch.html
    html_path = html + str(num) + '.html'

    ch.pyMuPDF_fitz(file, pdf_img, html_path)

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
    # main()
    app.run('0.0.0.0', port=50088)

