# -*- coding: utf-8 -*-
import io
import requests


def download_pdf(save_path, pdf_name, pdf_url):
    send_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
        "Connection": "keep-alive",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8"}
    response = requests.get(pdf_url, headers=send_headers)
    bytes_io = io.BytesIO(response.content)
    with open(save_path + "%s.pdf" % pdf_name, mode='wb') as f:
        f.write(bytes_io.getvalue())
        print('%s.PDF,下载成功！' % (pdf_name))


if __name__ == '__main__':
    save_path = ''
    pdf_url = "http://static.cninfo.com.cn/finalpage/2008-03-27/38334292.PDF"
    pdf_name = pdf_url.split('/')[-1].split('.')[0]
    download_pdf(save_path, pdf_name, pdf_url)





