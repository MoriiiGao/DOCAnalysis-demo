

def remove_sidebar(axis_list):
    """对左侧边栏进行删除"""
    #规避文章中的侧边栏
    axis_list.sort(key=lambda b:b[1][0][0])

    x_axis = list()
    for i in range(len(axis_list)):
        x_axis.append((axis_list[i][1][0][0], i))

    x_sum = 0
    for (x, i) in x_axis:
        x_sum += x
    x_average = x_sum / len(x_axis)

    left_x = list()
    left_x_axis = list()