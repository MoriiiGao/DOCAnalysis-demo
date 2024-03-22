import fitz

def mupdf(pdf_path):
    doc = fitz.open(pdf_path)
    for pg in range(doc.pageCount):
        page = doc[pg]
        text = page.extractBLOCKS
        print(text)
        #text_blocks = page.get_text('blocks')
        #for i in range(len(text_blocks)):
           # print(text_blocks[i])


if __name__ == '__main__':
    pdf_path = 'pdftest/test_3.pdf'
    mupdf(pdf_path)
