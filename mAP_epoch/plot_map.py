import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#导入实验数据
# dfSSD = pd.read_csv(r'SSD.csv')
# dfFasterrcnn = pd.read_csv(r'Fasterrcnn.csv')
# dfMaskrcnn = pd.read_csv(r'maskrcnn.csv')
dfYolo = pd.read_csv(r'YoloV5.csv')
dfours = pd.read_csv(r'ours1.csv')

"""
out = Index(['               epoch', '      train/box_loss', '      train/obj_loss',
       '      train/cls_loss', '   metrics/precision', '      metrics/recall',
       '     metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', '        val/box_loss',
       '        val/obj_loss', '        val/cls_loss', '               x/lr0',
       '               x/lr1', '               x/lr2'],
      dtype='object')
"""

# dfSSD = dfSSD['metrics/mAP_0.5:0.95'][:]
# dfFasterrcnn = dfFasterrcnn['metrics/mAP_0.5:0.95'][:]
# dfMaskrcnn = dfMaskrcnn['metrics/mAP_0.5:0.95'][:]
dfYolo = dfYolo['metrics/mAP_0.5:0.95'][:]
dfours = dfours['metrics/mAP_0.5:0.95'][:]


plt.figure(figsize=(10, 8), dpi=600)

x = [i for i in range(0, 150)]
# y1 = dfSSD
# y2 = dfFasterrcnn
# y3 = dfMaskrcnn
y4 = dfYolo
y5 = dfours

#plt.title('各模型mAP@0.5曲线')  # 标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.rcParams['axes.unicode_minus'] =False

plt.xlabel('Training Epochs', fontsize=20)  # x轴标题以及标题大小设置
plt.ylabel('mAP@0.5:0.95', fontsize=20)  # y轴标题
#刻度值字体大小设置（x轴和y轴同时设置）
plt.tick_params(labelsize=15)

# plt.plot(x, y1)  # 绘制折线图
# plt.plot(x, y2)  # 绘制折线图
# plt.plot(x, y3)  # 绘制折线图
plt.plot(x, y4)  # 绘制折线图
plt.plot(x, y5)  # 绘制折线图

# 设置曲线名称
# plt.legend(['SSD', 'Faster-RCNN', 'Mask-RCNN', 'YOLOV5', 'YOLOLayout(ours)'], loc=0, fontsize='xx-large')
plt.legend(['YOLOV5', 'YOLOLayout(ours)'], loc=0, fontsize='xx-large')
# plt.legend(['SSD', 'Faster-RCNN', 'Mask-RCNN', 'YOLOV5', 'ours'], loc=0, fontsize='xx-large')
#图例大小可选----'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'

plt.savefig('mAP@0.5:0.95.png', bbox_inches='tight',pad_inches=0) #保存图片，这里增加这两个参数可以消除保存下来图像的白边节省空间，bbox_inches='tight',pad_inches=0)
plt.show()  # 显示曲线图

