import os
import config
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np


def reorder_vertexes(xy_list):  # TODO 待优化
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:  # h横坐标相同
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index  # 使用first_v 记录下标
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index

    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))  # 计算斜率
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
               / (xy_list[index, 0] - xy_list[first_v, 0] + 0.000001)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]  # 第三个点已经判别出来了
    others.remove(third_v)
    # determine the second point which on the bigger side of the middle line
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]  # kx+b 的b
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index  # 这样就是顺时针
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
            xy_list[second_v, 0] - xy_list[fourth_v, 0] + 0.000001)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


# 缩短边的长度
def shrink_edge(xy_list, new_xy_list, edge, r, theta, ratio=config.shrink_ratio):
    if ratio == 0.0:
        return
    start_point = edge
    end_point = (edge + 1) % 4
    long_start_sign_x = np.sign(
        xy_list[end_point, 0] - xy_list[start_point, 0])
    new_xy_list[start_point, 0] = \
        xy_list[start_point, 0] + \
        long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])
    long_start_sign_y = np.sign(
        xy_list[end_point, 1] - xy_list[start_point, 1])
    new_xy_list[start_point, 1] = \
        xy_list[start_point, 1] + \
        long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])
    # long edge one, end point
    long_end_sign_x = -1 * long_start_sign_x
    new_xy_list[end_point, 0] = \
        xy_list[end_point, 0] + \
        long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])
    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = \
        xy_list[end_point, 1] + \
        long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])


# 进行坐标的收缩
def shrink(xy_list, ratio=config.shrink_ratio):
    if ratio == 0.0:
        return xy_list, xy_list
    # 按顺时针依次计算 相邻两个点的差

    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    # 计算欧氏距离
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))
    # determine which are long or short edges
    # long edge 为0 代表 边v1-v2, v3-v4 长
    # long edge 为0 代表 边v2-v3, v4-v1 长
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
    short_edge = 1 - long_edge
    # cal r length array    # TODO 这里跟论文不一样啊
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
    # r = [np.minimum(dis[i], dis[(i + 3) % 4]) for i in range(4)]
    # cal theta array
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += 0.000001
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
    # shrink two long edges
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)
    # shrink two short edges
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)
    # temp_new_xy_list 为缩了长边后的形状
    # new_xy_list为缩了短边后的形状
    # long_edge 为0或1，标志长边
    return temp_new_xy_list, new_xy_list, long_edge


def preprocess():
    """
    预处理：1.大小缩放，2.画出四边形，3.画出缩小后的四边形，4.保存按照片比例缩放后的gt,5.保存画好的照片到一个目录
    :return:
    """
    origin_image_dir = os.path.join(config.data_root, config.train_dir)
    origin_txt_dir = os.path.join(config.data_root, config.train_gt_dir)
    train_label_dir = os.path.join(config.data_root, config.train_label_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    show_gt_image_dir = os.path.join(config.data_root, config.show_gt_image_dir_name)
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    show_act_image_dir = os.path.join(config.data_root, config.show_act_image_dir_name)
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):
        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:
            # 更改大小RGB
            img_height, img_width = config.img_height, config.img_width
            scale_ratio_h = img_height / im.height
            scale_ratio_w = img_width / im.width

            # 这里需要注意确实宽和高的顺序不一样
            im = im.resize((img_width, img_height), Image.NEAREST).convert('RGB')

            show_gt_im = im.copy()
            # draw on the img
            draw = ImageDraw.Draw(show_gt_im)
            with open(os.path.join(origin_txt_dir,
                                   "gt_" + o_img_fname[:-4] + '.txt'), 'r', encoding="utf-8-sig") as f:
                anno_list = f.readlines()
            xy_list_array = np.zeros((len(anno_list), 4, 2), dtype=float)
            for anno, i in zip(anno_list, range(len(anno_list))):
                anno_colums = [int(num_str) for num_str in anno.strip().split(',')[:8]]
                anno_array = np.array(anno_colums)
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                # 缩放，变换坐标
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = reorder_vertexes(xy_list)
                xy_list_array[i] = xy_list
                _, shrink_xy_list, _ = shrink(xy_list, config.shrink_ratio)  # 论文中是0.3 但是davanced 用了0.2

                # 画出gt的四边形：
                draw.line(
                    [tuple(xy_list[0]), tuple(xy_list[1]), tuple(xy_list[2]), tuple(xy_list[3]), tuple(xy_list[0])],
                    width=2, fill='green')
                # 画出缩小后的四边形：
                draw.line([tuple(shrink_xy_list[0]),
                           tuple(shrink_xy_list[1]),
                           tuple(shrink_xy_list[2]),
                           tuple(shrink_xy_list[3]),
                           tuple(shrink_xy_list[0])
                           ],
                          width=2, fill='blue')
                # 保存训练数据的点
                np.save(os.path.join(train_label_dir, o_img_fname[:-4] + '.npy'), xy_list_array)

                # 保存画好的点
                show_gt_im.save(os.path.join(show_gt_image_dir, o_img_fname))

    train_img_list = os.listdir(origin_image_dir)
    print('found %d train images.' % len(origin_image_dir))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))


if __name__ == "__main__":
    preprocess()
