import base64
import sys

import fitz
import os
import time
from pathlib import Path
import torch
import cv2
import xlwt
from paddleocr import PPStructure
from paddleocr.ppstructure.table.predict_table import to_excel
from utils.plots import Annotator, colors
FILE = Path(__file__).resolve()
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device
from flask import Flask, request, send_from_directory

app = Flask(__name__)
table_engine = PPStructure(show_log=True)

if getattr(sys, 'frozen', False):
    absPath = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    absPath = os.path.dirname(os.path.abspath(__file__))

# args = init_args().parse_args(args=[])
# args.det_model_dir='inference/ch_PP-OCRv2_det_infer'
# args.rec_model_dir='inference/ch_PP-OCRv2_rec_infer'
# args.table_model_dir='inference/en_ppocr_mobile_v2.0_table_structure_infer'
# args.image_dir= 'pdftest/ch_2_images_0_fig_0.png'
# args.rec_char_dict_path='/home/hello/anaconda3/envs/gzc_torch/lib/python3.7/site-packages/paddleocr/ppocr/utils/ppocr_keys_v1.txt'
# args.table_char_dict_path='/home/hello/anaconda3/envs/gzc_torch/lib/python3.7/site-packages/paddleocr/ppocr/utils/dict/table_structure_dict.txt'
# args.det_limit_side_len=736
# args.det_limit_type='min'
# args.output='output/table'
# args.use_gpu=False
def tabel_rec_test():
    """调试代码"""
    img_path = 'table_test/table8.PNG'
    img = cv2.imread(img_path)

    #type为table可生成html type为figure可生成html
    result = table_engine(img)
    # print('result:', result)
    if result[0]['type'] == 'table':
        html_str = result[0]['res']['html']
        print("html:", html_str)
        to_excel(html_str, '2.xls')
    else:
        print('no table')
        # res = result[0]['res']
        # write_to_excel(res)
    # save_structure_res(result, save_dir, '1')


def write_to_excel(res, excel_path):
    """
    对无法用ppstructure生成excel的像素表格 通过简单判断写入excel(可优)
    通过文本块两个y值 来判断text对应的excel中的行数
    """
    i = 0#行
    j = 0#列
    line_symbol = list()
    new_workbook = xlwt.Workbook()
    sheet = new_workbook.add_sheet('table1')
    for idx, text_dict in enumerate(res):
        text = text_dict['text']
        axis = text_dict['text_region']
        print(text, axis)
        if idx == 0:
            line_symbol.append(axis[0][1])
            line_symbol.append(axis[-1][-1])
            sheet.write(i, j, text)
        else:
            if line_symbol[0] - 3 <= axis[0][1] <= line_symbol[0] + 3 and line_symbol[1] - 3 <= axis[-1][-1] <= line_symbol[1] + 3:
                j += 1
                sheet.write(i, j, text)
            else:
                i += 1
                j = 0
                sheet.write(i, j, text)
                line_symbol = []
                line_symbol.append(axis[0][1])
                line_symbol.append(axis[-1][-1])

    new_workbook.save(excel_path)

def table_rec(table_pix, path):
    table_name_list = []
    for idx, i in enumerate(table_pix):
        result = table_engine(i)
        table_name = '{}.xls'.format(idx)
        excel_path = path + table_name
        if result[0]['type'] == 'table':
            html_str = result[0]['res']['html']
            print(html_str)
            to_excel(html_str, excel_path)
        else:
            res = result[0]['res']
            write_to_excel(res, excel_path)
        table_name_list.append(table_name)

    return table_name_list

def table_rec1(table_pix, name, path):
    result = table_engine(table_pix)
    table_name = '{}.xls'.format(name)
    excel_path = path + table_name
    if result[0]['type'] == 'table':
        html_str = result[0]['res']['html']
        to_excel(html_str, excel_path)
    else:
        res = result[0]['res']
        write_to_excel(res, excel_path)

    return table_name

class TABLEDetct():
    def __init__(self, weights='/home/hello/gzc/3.project/MSLayout/runs/train/exp3/weights/best.pt',  # model.pt path(s)
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
        super(TABLEDetct, self).__init__()

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = imgsz  # check image size
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

        self.upload_file_path = './static/table_detect/ch_pdf/upload_file/' + self.time + '/'
        self.upload_file_to_img_path = './static/table_detect/ch_pdf/pdf_to_img/' + self.time + '/'
        self.layout_img = './static/table_detect/ch_pdf/layout/' + self.time + '/'
        self.recog_tab_path = './static/table_detect/ch_pdf/recog_tab/' + self.time + '/'
        self.excel_path = './static/table_detect/ch_pdf/recog_tab/' + self.time + '/'


        self.make_output(self.upload_file_path)
        self.make_output(self.upload_file_to_img_path)
        self.make_output(self.layout_img)
        self.make_output(self.recog_tab_path)
        self.make_output(self.excel_path)

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

    # def CutImage(self, source, axis_list):
    #     """
    #     根据坐标裁剪对应位置图像
    #     inputparam:axis_list--[(axis, label),...] 坐标+标签
    #     return:CutImageAddressPath--[(cutimage, label),...]
    #     """
    #     #存放截取图像地址的列表
    #     CutTablePix = list()
    #     filename = source.split('/')[-1].split('.')[0]
    #     image = cv2.imread(source)
    #     for idx, i in enumerate(axis_list):
    #         axis = i
    #
    #         tab_path = self.recog_tab_path + '/' + filename + "_fig_{}.jpg".format(idx)
    #         dst_img = image[axis[1]:axis[3], axis[0]:axis[2]]
    #         cv2.imwrite(tab_path, dst_img)
    #         CutTablePix.append(dst_img)
    #
    #     return CutTablePix

    def CutImage(self, source, axis, idx):
        filename = source.split('/')[-1].split('.')[0]
        image = cv2.imread(source)

        tab_path = self.recog_tab_path + '/' + filename + "_fig_{}.jpg".format(idx)
        dst_img = image[axis[1]:axis[3], axis[0]:axis[2]]
        cv2.imwrite(tab_path, dst_img)

        return dst_img

    def YOLODetect(self, img_path):

        source = str(img_path)
        name = source.split('/')[-1].split('.')[1]
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        TablePixList = list()  # 存放像素表格
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

                    for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                        if self.view_img:  # Add bbox to image
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            label = self.names[int(cls)]
                            axis = (x1, y1, x2, y2)

                            annotator.box_label(xyxy, label, color=colors(int(cls), True))

                            if label == "Table":
                                #获取表格像素
                                TablePix = self.CutImage(source, axis, idx)
                                TablePixList.append(TablePix)

                im0 = annotator.result()
                cv2.imwrite(self.layout_img + '/' + name + '.png', im0)

        return TablePixList

    def get_ori_table(self, PdfPath, ImagePath):
        """
        像素表格转base64
        """
        pdf_name = PdfPath.split('/')[-1].split('.')[0]
        pdfDoc = fitz.open(PdfPath)

        Table_Pix = list()#存放多页PDF解析后的像素表格
        for pg in range(pdfDoc.pageCount):
            page = pdfDoc[pg]
            #-------1.pdf转图片per-----------
            #获取像素
            rotate = int(0)
            zoom_x = 1.0
            zoom_y = 1.0
            mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pix = page.getPixmap(matrix=mat)

            #转存文档图像路径
            if not os.path.exists(ImagePath):
                os.makedirs(ImagePath)

            #保存文档像素
            img_path = ImagePath + '/' + pdf_name + '_' + 'images_%s.png' % pg
            pix.writePNG(img_path) #将图片写入指定的文件夹内

            #-------2.表格检测(dla)-----------
            TablePixList = self.YOLODetect(img_path)
            if len(TablePixList) != 0:
                for tp in TablePixList:
                    tab_base64 = self.np_to_base64(tp)
                    Table_Pix.append((pg, tab_base64))
            else:
                continue

        return Table_Pix

    def table_analysis(self, PdfPath, ImagePath):
        '''
        表格解析
        '''
        pdf_name = PdfPath.split('/')[-1].split('.')[0]
        pdfDoc = fitz.open(PdfPath)

        Table_Name = list()#存放多页PDF解析后生成的多个excel 的名字
        for pg in range(pdfDoc.pageCount):
            page = pdfDoc[pg]
            #-------1.pdf转图片per-----------
            #获取像素
            rotate = int(0)
            zoom_x = 1.0
            zoom_y = 1.0
            mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pix = page.getPixmap(matrix=mat)

            #转存文档图像路径
            if not os.path.exists(ImagePath):
                os.makedirs(ImagePath)

            #保存文档像素
            img_path = ImagePath + '/' + pdf_name + '_' + 'images_%s.png' % pg
            pix.writePNG(img_path) #将图片写入指定的文件夹内

            #-------2.表格检测(dla)-----------
            TablePixList = self.YOLODetect(img_path)
            #-------3.表格预测(ocr)-----------
            if len(TablePixList) != 0:
                table_name = table_rec(TablePixList, self.excel_path)
                for tb in table_name:
                    Table_Name.append(tb)
            else:
                continue

        return Table_Name

def table_test():
    pdf_path = 'pdftest/ch_2.pdf'
    tb_det = TABLEDetct()
    pdf_convert_img_path = tb_det.upload_file_to_img_path
    Table_Pix = tb_det.get_ori_table(pdf_path, pdf_convert_img_path)
    Table_name = tb_det.table_analysis(pdf_path, pdf_convert_img_path)
    print("Table_pix:", Table_Pix)
    print("Table_name:", Table_name)

#http://39.104.86.105:8014/getimg
@app.route('/getimg', methods=['GET', 'POST'])
def process_getimg():
    file = request.files.get('PDFFile')
    filename = file.filename.split('.')[0]
    filetype = file.filename.split('.')[1]

    tb_det = TABLEDetct()
    pdfpath = tb_det.upload_file_path + filename + '.' + filetype
    pdf_img = tb_det.upload_file_to_img_path
    file.save(pdfpath)

    Table_Pix = tb_det.get_ori_table(pdfpath, pdf_img)
    print(Table_Pix)
    try:
        if len(Table_Pix) == 1:#如果只解析了一个表格
            data = {"code": 200, "tab": Table_Pix[0]}
        else:                  #多个表格
            data = {"code": 200}
            for i in range(len(Table_Pix)):
                data['tab{}'.format(i)] = Table_Pix[i]
    except Exception as e:
        print(e)
        data = {'code': 500, 'msg': 'Error', 'data': 'Table Parsing Failed'}
    return data

#http://39.104.86.105:8014/getexcel
@app.route('/getexcel', methods=['GET', 'POST'])
def process_getexcel():
    img = request.files.get('TabImg')
    filename = img.filename.split('.')[0]
    tabpix = cv2.imread(img)

    tb_det = TABLEDetct()
    excel_path = tb_det.excel_path
    table_name = table_rec1(tabpix, filename, excel_path)
    print(os.path.join(excel_path, table_name))
    return send_from_directory(excel_path, table_name)

#http://39.104.86.105:8014/getexcel
# @app.route('/getexcel', methods=['GET', 'POST'])
# def process_getexcel():
#     file = request.files.get('PDFFile')
#     filename = file.filename.split('.')[0]
#     filetype = file.filename.split('.')[1]
#
#     tb_det = TABLEDetct()
#     pdfpath = tb_det.upload_file_path + filename + '.' + filetype
#     pdf_img = tb_det.upload_file_to_img_path
#     excel_path = tb_det.excel_path
#     file.save(pdfpath)
#
#     table_name = tb_det.table_analysis(pdfpath, pdf_img)
#     for i in table_name:
#         print(os.path.join(excel_path, i))
#     dl_name = '{}.zip'.format(filename)
#     if len(table_name) == 1:  # 如果只有一个表格 直接下载
#         return send_from_directory(excel_path, table_name[0])
#     else:  # 如果有多个表格 打包成zip通过BytesIO直接写入内存中
#         memory_file = BytesIO()
#         with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
#             for na in table_name:
#                 with open(os.path.join(excel_path, na), 'rb') as fp:
#                     zf.writestr(na, fp.read())
#         memory_file.seek(0)
#         return send_file(memory_file, attachment_filename=dl_name, as_attachment=True)


if __name__ == "__main__":
    tabel_rec_test()
    # table_test()
    # app.run('0.0.0.0', port=10086)
