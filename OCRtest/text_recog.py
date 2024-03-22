import cv2
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")


def border_extension(img):
    """
    问题：paddleocr无法精准识别小图和长图
    方法：对图片进行预处理 增加图片的大小 给图片加一定大小的框
    对裁剪出来的小图和长图 在边界填充一定数目的像素
    """
    h, w, c = img.shape[:]
    border = [0, 0]
    transform_size = 320  # 图片增加边框到320大小
    if w < transform_size or h < transform_size:
        if h < transform_size:
            border[0] = (transform_size - h) / 2.0
        if w < transform_size:
            border[0] = (transform_size - 2) / 2.0

        # top, buttom, left, right对应边界的像素数目
        img = cv2.copyMakeBorder(img, int(border[0]), int(border[0]), int(border[1]), int(border[1]),
                                 cv2.BORDER_CONSTANT,
                                 value=[215, 215, 215])
    return img

def OCR(dst_img):
    img = border_extension(dst_img)
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

if __name__ == '__main__':
    img = '13.jpg'
    im = cv2.imread(img)
    text = OCR(im)
    print(text)