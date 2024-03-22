import os
import cv2
from docx.oxml.ns import qn
from pdf2docx import Converter
from docx import Document
import time
from typing import List
import os
import subprocess
import pypandoc
from paddleocr import PPStructure, save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

basepath = os.path.dirname(__file__)


def make_output(path: List):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
    return True

class Format_converion_pdf2docx():
    def __init__(self):

        self.url = 'http://172.16.16.112:8888/'
        self.time = time.strftime("%Y-%m-%d-%H-%M-%S")

        self.upload_file_path = './static/AnalysisSystem/format_converion/docx/upload_file/' + self.time + '/'
        self.word_path = './static/AnalysisSystem/format_converion/docx/word/' + self.time + '/'

        self.upload_file_to_img_path = './static/AnalysisSystem/format_converion/docx/pdf_to_img/' + self.time + '/'
        self.layout_img = './static/AnalysisSystem/format_converion/html/layout/' + self.time + '/'
        self.recog_fig_path = './static/AnalysisSystem/format_converion/html/recog_fig/' + self.time + '/'
        self.recog_tab_path = './static/AnalysisSystem/format_converion/html/recog_tab/' + self.time + '/'
        self.recog_equa_path = './static/AnalysisSystem/format_converion/html/recog_equa/' + self.time + '/'
        self.detect_path = './static/AnalysisSystem/format_converion/html/detet/' + self.time + '/'

        make_output([self.upload_file_path, self.word_path, self.upload_file_to_img_path, self.layout_img,
                     self.recog_fig_path, self.recog_tab_path, self.recog_equa_path, self.detect_path])

    def Modify_font(self, doc):
        for para in doc.paragraphs:
            for run in para.runs:
                run.font.name = "宋体"

        return doc

    def main1(self, image_path, docpath):
        # 中文测试图
        table_engine = PPStructure(recovery=True)
        # 英文测试图
        # table_engine = PPStructure(recovery=True, lang='en')

        save_folder = docpath
        img_path = image_path
        img = cv2.imread(img_path)
        #获取解析的表格
        result = table_engine(img)
        save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

        for res_ in result:
            if isinstance(res_, list):
                for line in res_:
                    line.pop('img')
                    print(line)
                # for line in result[0]:
                #     line.pop('img')
                #     print(line)

                h, w, _ = img.shape
                #组件排序
                res = sorted_layout_boxes(res_, w)
                convert_info_docx(img, res, save_folder, os.path.basename(img_path).split('.')[0])
                filepath = basepath + self.word_path.split('.')[-1]

                return filepath

    def main(self, PDFPath, PDFName):

        #convert pdf2docx
        cv = Converter(PDFPath)
        DOCX = self.word_path + PDFName + '.docx'
        cv.convert(self.word_path + PDFName + '.docx', start=0, end=None)
        cv.close()
        #修改字体
        # DOC = Document(DOCX)
        # DOC.styles['Normal'].font.name = 'Times New Roman'
        # DOC.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
        # DOC = self.Modify_font(DOC)
        # DOC.save(DOCX)

        filepath = basepath + self.word_path.split('.')[-1]

        return filepath

    def test_unoconv(self, DOCPath, PDFPath):
        os.system('unoconv -f docx {} {}'.format(DOCPath, PDFPath))

    def test_pandoc(self, pdfpath, docpath):
        name = pdfpath.split('/')[-1].split('.')[0]
        output = pypandoc.convert_file(pdfpath, 'docx', outputfile=docpath + name + '.docx')

    def test(self, PDFPath, DOCPath):
        # subprocess.run(['libreoffice', '--headless', '--convert-to', 'docx', '--outdir', DOCPath, PDFPath],
        #                stdout=subprocess.PIPE)
        subprocess.run(['libreoffice', '--headless', '--convert-to', 'docx', '--outdir', DOCPath, PDFPath],
                       stdout=subprocess.PIPE)

        if os.path.exists(DOCPath):
            print(f"PDF文件已经打印成功转换为:{DOCPath}")
        else:
            print("PDF文件转换失败")


if __name__ == "__main__":
    basepath = os.path.dirname(__file__)
    pdfpath = str(basepath) + '/test/pdf2html/ch_1.pdf'
    image = str(basepath) + '/pdftest/ch_2_00.png'
    doc = Format_converion_pdf2docx()
    docpath = doc.word_path
    # doc.test_unoconv(docpath, pdfpath)
    # doc.test(pdfpath, docpath)
    # doc.test_pandoc(pdfpath, docpath)
    # doc.main(pdfpath, 'ch_1')
    doc.main1(image, docpath)