import fitz


def pdf_extract(PdfPath):
    '''
    :param PdfPath:
    :return:
    '''
    pdfDoc = fitz.open(PdfPath)
    for pg in range(pdfDoc.pageCount):
        page = pdfDoc[pg]

        text_blocks = page.get_text('blocks')
        text_blocks.sort(key=lambda b: (b[0], b[1]))

        for b in text_blocks:
            text = b[4]
            print(text)

if __name__ == '__main__':
    pdf = 'ch_1.pdf'
    pdf_extract(pdf)