from flask import Flask, request, jsonify, send_from_directory

from FIleBlocksRecognition import BlocksRecognition
from FileLayoutAnalysis import LayoutAnalysis
from Format_Converion_pdf2html import Format_converion_pdf2html
from TableAnalysis import TableAnalysis
from Format_Converion_pdf2docx import Format_converion_pdf2docx
import random

app = Flask(__name__)
excel_path_dict = list()
excel_name_dict = list()
# table_file_dict = list()

#文档布局分析 http://192.168.100.248:8888/LayoutAnalysis
@app.route('/DOC/LayoutAnalysis', methods=['GET', 'POST'])
def LayoutAnalysis_():
    file = request.files.get('FileAnalysis')
    filename = file.filename.split('.')[0]
    filetype = file.filename.split('.')[1]

    Lay = LayoutAnalysis()
    SaveFilePath = Lay.upload_file_path + filename + '.' + filetype
    file.save(SaveFilePath)

    try:
        data = Lay.pdf2img(SaveFilePath)
    except Exception as e:
        print(e)
        data = {'code': 500, 'msg': 'PDF Parsing Failed', 'error': e}
    return jsonify(data)

@app.route('/DOC/TableAnalysis', methods=['GET', 'POST'])
def TableAnalysis_():
    global excel_path_dict
    global excel_name_dict
    excel_path_dict = list()
    excel_name_dict = list()

    file = request.files.get('TableAnalysis')
    filename = file.filename.split('.')[0]
    filetype = file.filename.split('.')[1]

    Tab = TableAnalysis()
    SaveFilePath = Tab.upload_file_path + filename + '.' + filetype
    file.save(SaveFilePath)

    try:
        data = Tab.main(SaveFilePath)
        for i in data['data']:
            if len(i['page_res']) != 0:
                for j in i['page_res']:
                    excel_path_dict.append(j['excel_path'])
                    excel_name_dict.append(j['excel_name'])
        # print(excel_path_dict)
        # print(excel_name_dict)
    except Exception as e:
        data = {'code': 500, 'msg': 'Table Parsing Failed', 'error': e}
    return jsonify(data)

#http://39.104.86.105:8018/getexcel/1
@app.route('/DOC/GetExcel', methods = ['GET', 'POST'])
def process_getexcel():
    # num = request.values.get('id')
    num = request.json['id']
    try:
        table_id = int(num)
        excel_path = excel_path_dict[table_id]
        excel_name = excel_name_dict[table_id]
        return send_from_directory(excel_path, excel_name)
    except Exception as e:
        print(e)
        return jsonify({'code': 500, 'msg': 'Table Parsing Failed', 'error': e})

#http://39.104.86.105:8018/PDF2HTML
@app.route('/DOC/PDF2HTML', methods=['GET', 'POST'])
def format_converion_pdf2html():
    try:
        file = request.files.get('pdf2html')
        file_name = file.filename.split('.')[0]
        file_type = file.filename.split('.')[1]

        CH = Format_converion_pdf2html()
        pdf_path = CH.upload_file_path#pdf保存地址
        pdf_abspath = pdf_path + file_name + '.' + file_type
        pdf_img = CH.upload_file_to_img_path
        file.save(pdf_abspath)

        html_path = CH.html
        html_name, HTMLstr = CH.pyMuPDF_fitz(pdf_abspath, pdf_img, html_path)
        data = {'code': 200, 'msg': 'success', 'HTMLName': html_name}

        # return send_from_directory(html_path, html_name)
        return send_from_directory(html_path, html_name)
    except Exception as e:
        print(e)
        data = {'code': 500, 'msg': e, 'data': 'PDF parsing failed'}
        return jsonify(data)

# http://39.104.86.105:8018/PDF2DOCX
@app.route('/DOC/PDF2DOCX', methods=['GET', 'POST'])
def format_convert_pdf2docx():
    try:
        file = request.files.get('pdf2docx')
        filename, filetype = file.filename.split('.')[0], file.filename.split('.')[-1]

        CVDOC = Format_converion_pdf2docx()
        PDFPath = CVDOC.upload_file_path
        PDFFile = PDFPath + filename + '.' + filetype
        file.save(PDFFile)

        DOCXPath = CVDOC.main(PDFFile, filename)
        return send_from_directory(DOCXPath, filename + '.docx')

    except Exception as e:
        data = {'code': 500, 'msg': 'error', 'data': e}
        return jsonify(data)
#http://39.104.86.105:8018/BlocksRecgnition
@app.route('/DOC/BlocksRecgnition', methods=['GET', 'POST'])
def FBlocksRecognition():
    try:
        file = request.files.get('FileBlocksRecog')
        file_name = file.filename.split('.')[0]
        file_type = file.filename.split('.')[1]

        FBR = BlocksRecognition()
        pdf_path = FBR.upload_file_path
        pdfsavepath = pdf_path + file_name + '.' + file_type
        file.save(pdfsavepath)

        data = FBR.main(pdfsavepath)
        return data
    except Exception as e:
        data = {'code': 500, 'msg': e, 'data': 'PDF Recog Failed'}
        return jsonify(data)


if __name__ == '__main__':
    app.run('0.0.0.0', port=8888)
