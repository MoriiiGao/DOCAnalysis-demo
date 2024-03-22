import pdfplumber

def pdf_extract(pdf):
    pdf = pdfplumber.open(pdf)
    page = pdf.pages[0]
    text = page.extract_text()
    # print(text)
    # for i in text:
    #     print(i)
    words = page.extract_words()
    for i in words:
        print(i)

if __name__ == '__main__':
    pdf = 'ch_1.pdf'
    pdf_extract(pdf)