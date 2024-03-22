def get_main_or_left_axis(bbox_blocks):
    '''
    获取整页布局第一个main或left类型div的x和y坐标
    '''
    main_padding_list = list()  # 存放第一个main或left类型div的x和y坐标
    for i, bbox_list in enumerate(bbox_blocks):
        for j, item in enumerate(bbox_list[0]):
            if (bbox_list[1] == 'main' or bbox_list[1] == 'left') and len(main_padding_list) == 0:
                main_padding_list.append(((item[1][0][0], item[1][0][1]), item[1][1], item[1][2]))
    return main_padding_list

def css_div(bbox_blocks, pdf_w):
        '''
        css样式模板
        :param bbox_list:
        :param pdf_w:
        :return:
        '''
        text_font_size = 14
        title_font_size = 20
        divstring = ''
        div_string = list()

        Max_left_block_w = list()
        Max_right_block_w = list()
        for bbox_list in bbox_blocks:
            if bbox_list[1] == 'left':
                for item in bbox_list[0]:
                    Max_left_block_w.append(item[1][1])

            if bbox_list[1] == 'right':
                for item in bbox_list[0]:
                    Max_right_block_w.append(item[1][1])

        if len(Max_left_block_w) != 0:
            left_w_max = max(Max_left_block_w)
        else:
            left_w_max = 0

        if len(Max_right_block_w) != 0:
            right_w_max = max(Max_right_block_w)
        else:
            right_w_max = 0

        # try:
        #     left_w_max = max(Max_left_block_w)#左边block中 所有宽度的最大值
        # except Exception as e:
        #     pass
        #
        # try:
        #     right_w_max = max(Max_right_block_w)
        # except Exception as e:
        #     pass

        main_padding_list = get_main_or_left_axis(bbox_blocks)
        # 进入大列表for循环

        for i, bbox_list in enumerate(bbox_blocks):
            divstring = ''
            #循环每一个小列表
            for j, item in enumerate(bbox_list[0]):
                if j != len(bbox_list[0])-1:
                    # margin = abs(bbox_list[0][j+1][1][0][1] - (item[1][0][1] + item[1][2]))
                    margin = 10
                else:
                    margin = 0

                if bbox_list[1] == 'main':#
                    if item[0][-1] == 'Title':
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;' \
                                 'float: left; '.format(100, title_font_size)#float(item[1][1]/pdf_w) * 100
                        divstring += '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1],cssstr,item[0][1])
                    elif item[0][-1] == 'Text':
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'float: left; '.format(100, text_font_size)
                        divstring += '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1], cssstr, item[0][1])

                    elif item[0][-1] == 'Figure':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'float: left; ' \
                                 'margin-bottom: {2}px'.format(
                                    item[1][1], item[1][2], margin
                            )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    elif item[0][-1] == 'Table':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'float: left; ' \
                                 'margin-bottom: {2}px'.format(
                                    item[1][1], item[1][2], margin
                            )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    # elif item[0][-1] == 'Figure':
                    #     cssstr = 'width:{0}px;' \
                    #              'height:{1}px; ' \
                    #              'float: left; ' \
                    #              'margin-bottom: {2}px;'\
                    #              'margin-left: {3}px'.format(
                    #                 item[1][1],item[1][2],margin,item[1][3] - item[1][0][0]
                    #         )
                    #     divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    # elif item[0][-1] == 'Table':
                    #     cssstr = 'width:{0}px;' \
                    #              'height:{1}px; ' \
                    #              'float: left; ' \
                    #              'margin-bottom: {2}px;' \
                    #              'margin-left: {3}px'.format(
                    #         item[1][1], item[1][2], margin, item[1][3] - item[1][0][0]
                    #     )
                    #     divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                elif bbox_list[1] == 'left' and left_w_max != 0:
                    if len(main_padding_list) == 0:
                        main_padding_list.append(((item[1][0][0], item[1][0][1]), item[1][1], item[1][2]))
                    if item[0][-1] == 'Title' and j == 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;'.format((float(item[1][1]) / left_w_max) * 100, title_font_size)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Text' and j == 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;'.format((float(item[1][1]) / left_w_max) * 100, text_font_size)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Title' and j != 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;' \
                                 'margin-bottom: {2}px'.format((float(item[1][1]) / left_w_max) * 100, title_font_size, margin)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Text' and j != 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'margin-bottom: {2}px'.format((float(item[1][1]) / left_w_max) * 100, text_font_size,margin)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Figure':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'margin-bottom: {2}px;'.format(item[1][1], item[1][2], margin)
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    elif item[0][-1] == 'Table':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'margin-bottom: {2}px;'.format(
                                    item[1][1], item[1][2], margin
                            )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                elif bbox_list[1] == 'right' and right_w_max != 0:
                    if item[0][-1] == 'Title' and j == 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;'.format((float(item[1][1]) / right_w_max) * 100, title_font_size)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr,item[0][1])

                    elif item[0][-1] == 'Text' and j == 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;'.format((float(item[1][1]) / right_w_max) * 100, text_font_size)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr,item[0][1])

                    elif item[0][-1] == 'Title' and j != 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'font-weight:bold;' \
                                 'margin-bottom: {2}px'.format((float(item[1][1]) / right_w_max) * 100, title_font_size,
                                                               margin)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr, item[0][1])

                    elif item[0][-1] == 'Text' and j != 0:
                        cssstr = 'width: {0}%;' \
                                 'height: auto;' \
                                 'font-size:{1}px;' \
                                 'font-family:SimSun;' \
                                 'margin-bottom: {2}px'.format((float(item[1][1]) / right_w_max) * 100, text_font_size,
                                                               margin)
                        divstring += '<div style="{0}">{1}</div>'.format(cssstr,item[0][1])

                    elif item[0][-1] == 'Figure':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'margin-bottom: {2}px;'.format(
                            item[1][1], item[1][2], margin
                        )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)

                    elif item[0][-1] == 'Table':
                        cssstr = 'width:{0}px;' \
                                 'height:{1}px; ' \
                                 'margin-bottom: {2}px;'.format(
                            item[1][1], item[1][2], margin
                        )
                        divstring += '<img src=data:image/png;base64,{0} style="{1}"/>'.format(item[0][0], cssstr)


            #padding-top计算main最后一个div与left或right第一个div的差值
            #main最后一个div的y坐标:main_padding_list[-1][0][1]
            #main最后一个div的高度h:main_padding_list[-1][2]
            #left第一个div的y坐标:bbox_list[0][0][1][2]
            #退出小模块for循环 main_padding_list[-1][1]
            #left and right padding_top:abs(bbox_list[0][0][1][2] - (main_padding_list[-1][0][1] + main_padding_list[-1][2]))
            if bbox_list[1] == 'left' and left_w_max != 0:
                cssstr = 'width: {0}%;' \
                         'float: left;' \
                         'padding-top:{1}px;' \
                         'box-sizing: border-box;'.format((left_w_max / (pdf_w - main_padding_list[0][0][0]*2))*100
                                                          , 10)
                divstring = '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1], cssstr, divstring)
            elif bbox_list[1] == 'right' and right_w_max != 0:
                cssstr = 'width: {0}%;' \
                         'float: right;' \
                         'padding-top:{1}px;' \
                         'box-sizing: border-box;'.format((right_w_max / (pdf_w - main_padding_list[0][0][0]*2))*100, 10)
                divstring = '<div class="{0}" style="{1}">{2}</div>'.format(bbox_list[1], cssstr, divstring)

            div_string.append(divstring)

        DIVsrting = ' '.join(div_string)
        #最外层加一个padding
        cssstr = 'width: {0}px;' \
                 'height: 100%;' \
                 'box-sizing: border-box;' \
                 'padding-left: {1}px;' \
                 'padding-top:{2}px;'\
                 'padding-right:{3}px'.format(pdf_w,
                main_padding_list[0][0][0],main_padding_list[0][0][1],main_padding_list[0][0][0])
        DIV_string = '<div  style="{0}">{1}</div><hr style="width:100%"/>'.format(cssstr, DIVsrting)

        return DIV_string

def three_bbox_sort(bbox_list, pdf_w):
        '''
        将bbox分为三种情况：单独一栏的[横跨两列]、在左列、在右列
        :param bbox_list:
        :return:
        '''
        left_blocks = list()
        left_x = list()
        right_blocks =list()
        right_x = list()
        for i in range(len(bbox_list)):
            if bbox_list[i][1][0][0] < pdf_w/2:
                left_x.append(bbox_list[i][1][0][0])
            elif bbox_list[i][1][0][0] > pdf_w/2:
                right_x.append(bbox_list[i][1][0][0])

        for i in range(len(bbox_list)):
            if bbox_list[i][1][0][0] < pdf_w/2:
                bbox_item = bbox_list[i]
                bbox_item_list = list(bbox_item)
                bbox_w_h = bbox_item_list[1]
                bbox_w_h_list = list(bbox_w_h)
                left_corner_axis = bbox_w_h_list[0]
                left_corner_axis_list = list(left_corner_axis)

                left_corner_axis_list[0] = min(left_x)

                left_corner_axis = tuple(left_corner_axis_list)
                bbox_w_h_list[0] = tuple(left_corner_axis)
                bbox_w_h = tuple(bbox_w_h_list)
                bbox_item_list[1] = bbox_w_h
                bbox_list[i] = tuple(bbox_item_list)
                left_blocks.append(bbox_list[i])

            elif bbox_list[i][1][0][0] > pdf_w/2:
                bbox_item = bbox_list[i]
                bbox_item_list = list(bbox_item)
                bbox_w_h = bbox_item_list[1]
                bbox_w_h_list = list(bbox_w_h)
                right_corner_axis = bbox_w_h_list[0]
                right_corner_axis_list = list(right_corner_axis)

                right_corner_axis_list[0] = min(right_x)

                right_corner_axis = tuple(right_corner_axis_list)
                bbox_w_h_list[0] = right_corner_axis
                bbox_w_h = tuple(bbox_w_h_list)
                bbox_item_list[1] = bbox_w_h
                bbox_list[i] = tuple(bbox_item_list)
                right_blocks.append(bbox_list[i])
        # ----------------------------------从左侧栏中筛选除中间栏--------------------------------------
        single_line_blocks = list()
        del_left_blocks = list()
        for i in range(len(left_blocks)):
            width = left_blocks[i][1][1]
            x = left_blocks[i][1][0][0]
            if x + width > (pdf_w / 2):#
            # if x + width > (pdf_w / 2) + 40:#
                single_line_blocks.append(left_blocks[i])
                del_left_blocks.append(i)

        del_left_blocks = del_left_blocks[::-1]
        for i in del_left_blocks:
            left_blocks.pop(i)

        #按y轴对每个列表中的bbox进行排序
        single_line_blocks.sort(key=lambda b: b[1][0][1])
        left_blocks.sort(key=lambda b: b[1][0][1])
        right_blocks.sort(key=lambda b: b[1][0][1])

        all_bbox_list = list()
        for single_block in single_line_blocks:
            all_bbox_list.append((single_block, 'main'))
        for left_block in left_blocks:
            all_bbox_list.append((left_block, 'left'))
        for right_block in right_blocks:
            all_bbox_list.append((right_block, 'right'))

        all_bbox_list.sort(key=lambda b: b[0][1][0][1])

        new_bbox_list = list()
        main_list = list()
        left_list = list()
        right_list = list()
        for bbox in all_bbox_list:
            if bbox[1] == 'left':
                left_list.append(bbox[0])
                if len(main_list) != 0:
                    new_bbox_list.append((main_list, 'main'))
                    main_list = list()

            elif bbox[1] == 'right':
                right_list.append(bbox[0])
                if len(main_list) != 0:
                    new_bbox_list.append((main_list, 'main'))
                    main_list = list()

            elif bbox[1] == 'main':#标识符为main的时候 将left和right分别存进一个列表 然后更新一个新列表
                main_list.append(bbox[0])
                if len(left_list) != 0:
                    new_bbox_list.append((left_list, 'left'))
                    left_list = list()
                if len(right_list) != 0:
                    new_bbox_list.append((right_list, 'right'))
                    right_list = list()

        #将保存的列表保存
        if len(main_list) != 0:
            new_bbox_list.append((main_list, 'main'))
        if len(left_list) != 0:
            new_bbox_list.append((left_list, 'left'))
        if len(right_list) != 0:
            new_bbox_list.append((right_list, 'right'))

        return new_bbox_list

def gen_css(text_axis_list, figure_axis_list, table_axis_list, title_axis_list, pdf_w):

        text_axis_list.sort(key=lambda b: (b[1][0][0], b[1][0][1]))
        if len(figure_axis_list) != 0 and len(table_axis_list) != 0 and len(title_axis_list) != 0:
            bbox_list = figure_axis_list + table_axis_list + text_axis_list + title_axis_list
        elif len(figure_axis_list) != 0 and len(title_axis_list) != 0 and len(table_axis_list) == 0:
            bbox_list = figure_axis_list + title_axis_list + text_axis_list
        elif len(figure_axis_list) != 0 and len(table_axis_list) != 0 and len(title_axis_list) == 0:
            bbox_list = figure_axis_list + table_axis_list + text_axis_list
        elif len(figure_axis_list) == 0 and len(table_axis_list) != 0 and len(title_axis_list) != 0:
            bbox_list = table_axis_list + title_axis_list + text_axis_list
        elif len(figure_axis_list) != 0 and len(table_axis_list) == 0 and len(title_axis_list) == 0:
            bbox_list = figure_axis_list + text_axis_list
        elif len(figure_axis_list) == 0 and len(table_axis_list) != 0 and len(title_axis_list) == 0:
            bbox_list = table_axis_list + text_axis_list
        elif len(figure_axis_list) == 0 and len(table_axis_list) == 0 and len(title_axis_list) != 0:
            bbox_list = title_axis_list + text_axis_list
        elif len(figure_axis_list) == 0 and len(table_axis_list) == 0 and len(title_axis_list) == 0:
            bbox_list = text_axis_list

        bbox_blocks = three_bbox_sort(bbox_list, pdf_w)
        divstring = css_div(bbox_blocks, pdf_w)

        return divstring