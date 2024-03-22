import cv2
import numpy as np

def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height):
    '''
    _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
    '''

    bitmap = _bitmap
    height, width = bitmap.shape

    # findContours 获取轮廓，如长方形获取四点顶点坐标
    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    # py2、py3 不同版本的情况
    if len(outs) == 3:
        img, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]

    # 文本框最大数量
    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        #  计算最小包围矩，获取四个坐标点，左上为起点（顺时针）
        points, sside = get_mini_boxes(contour)
        # 长方形中宽高最小值过滤
        if sside < min_size:
            continue
        points = np.array(points)
        # 利用 points 内部预测概率值，计算出一个score,作为实例的预测概率
        score = box_score_fast(pred, points.reshape(-1, 2))
        # score 得分的过滤
        if box_thresh > score:
            continue
        # shrink反向还原，之前概率图进行了缩放，故还原
        box = unclip(points).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box)

        # 还原到原始坐标，反向还原之后，还需要还原到原始图片（原始图片在预处理时被缩放处理）
        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.astype(np.int16))
        scores.append(score)
    return np.array(boxes, dtype=np.int16), scores


def get_mini_boxes(contour):
    # 返回点集 cnt 的最小外接矩形：# 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    bounding_box = cv2.minAreaRect(contour)
    # 排序，最终以左上的坐标为起点，顺时针排列四个坐标点
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])