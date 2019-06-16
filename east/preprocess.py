# 文件的作用
# 1.分割训练集、验证集和测试集，形成info信息
# 2.生成训练集、验证集和测试集 包括 img、gt
# 3.生成训练集、验证集和测试集的可视化图片信息

import os
import east.config as config
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import time


def split_train_valid_to_info(train_valid_dir, train_info_file, valid_info_file, ratio=0.1):
    """
    分割训练街和测试集id 到两个不同的info文件中
    :param train_valid_dir:
    :param train_info_file:
    :param valid_info_file:
    :param ratio: 测试集占比
    :return:
    """
    with open(train_info_file, 'w', encoding='utf-8') as t_f:
        with open(valid_info_file, 'w', encoding='utf-8') as v_f:
            train_count = 0
            valid_count = 0
            file_list = os.listdir(train_valid_dir)
            for filename, _ in zip(file_list, tqdm(range(len(file_list)), desc="创建train.info和valid.info")):
                if valid_count < ratio * (config.num_train_samples + config.num_valid_samples):
                    valid_count += 1
                    v_f.write("valid_number:" + str(valid_count) + " " + filename.split(".")[0] + "\n")
                    time.sleep(0.001)
                else:
                    train_count += 1
                    t_f.write("train_number:" + str(train_count) + " " + filename.split(".")[0] + "\n")
                    time.sleep(0.001)


def create_test_info(original_test_img_dir, test_info_file):
    """
    创建test.info
    :param original_test_img_dir:
    :param test_info_file:
    :return:
    """
    with open(test_info_file, "w", encoding="utf-8") as t_f:
        test_count = 0
        file_list = os.listdir(original_test_img_dir)
        for filename, _ in zip(file_list, tqdm(range(len(file_list)), desc="创建test.info")):
            test_count += 1
            t_f.write("test_number:" + str(test_count) + " " + filename.split(".")[0] + "\n")
            time.sleep(0.001)


# 第一步，分割训练集、验证集和测试集，形成info信息
def create_info_file():
    # 参数
    original_train_img_dir = config.original_train_img_dir
    original_test_img_dir = config.original_test_img_dir
    train_info_file = config.train_info_file
    valid_info_file = config.valid_info_file
    test_info_file = config.test_info_file
    # 调用
    split_train_valid_to_info(original_train_img_dir, train_info_file, valid_info_file)
    create_test_info(original_test_img_dir, test_info_file)


def create_img_dir(info_file, original_img_dir, img_dir, img_height, img_weight):
    """
    创建缩放后的用于训练的图片
    :param info_file:
    :param original_img_dir:
    :param img_dir:
    :param img_height:
    :param img_weight:
    :return:
    """
    with open(info_file, "r", encoding="utf-8") as info_file:
        lines = info_file.readlines()
        file_id_list = [line.strip().split(" ")[1] for line in lines]
        for file_id, _ in zip(file_id_list, tqdm(range(len(file_id_list)), desc="创建训练or验证or测试图片")):
            jpg_img_name = os.path.join(original_img_dir, file_id + ".jpg")
            assert os.path.exists(jpg_img_name), "no image file found by file_id：%s" % file_id
            img = Image.open(jpg_img_name)
            img = img.resize((img_weight, img_height), Image.NEAREST).convert('RGB')
            img = np.array(img)
            img = 255 - img
            img = Image.fromarray(img)
            img_save_name = os.path.join(img_dir, file_id + "_input.jpg")
            img.save(img_save_name)


def reorder_vertexes(xy_list):
    """
     将四边形坐标点重新进行排序，顺时针顺序
     思想：计算重心和四个点形成角的正切，
    :param xy_list:
    :return:
    """
    centroid_coord = xy_list.sum(axis=0) / xy_list.shape[0]
    delta_xy = xy_list - centroid_coord
    pos_half_pai_index = -1
    neg_half_pai_index = -1
    left_indexs = []
    right_indexs = []
    res_index = []
    for i in range(delta_xy.shape[0]):
        if delta_xy[i][0] == 0 and delta_xy[i][1] > 0:
            pos_half_pai_index = i
        elif delta_xy[i][0] == 0 and delta_xy[i][1] < 0:
            neg_half_pai_index = i
        else:
            if delta_xy[i][0] < 0:
                left_indexs.append([delta_xy[i][1] / delta_xy[i][0], i])
            else:
                right_indexs.append([delta_xy[i][1] / delta_xy[i][0], i])
    left_indexs.sort(key=lambda x: x[0], reverse=True)
    right_indexs.sort(key=lambda x: x[0], reverse=True)
    if pos_half_pai_index != -1:
        res_index.append(pos_half_pai_index)
    res_index += np.array(right_indexs)[:, 1].tolist()
    if neg_half_pai_index != -1:
        res_index.append(neg_half_pai_index)
    res_index += np.array(left_indexs)[:, 1].tolist()
    res_index = [int(index) for index in res_index]
    res = np.zeros(shape=xy_list.shape, dtype=np.float)
    for i in range(len(res_index)):
        res[i] = xy_list[res_index[i]]
    return res


def shrink(xy_list, ratio):
    """
    对四边形进行收缩
    思想：找到长边和短边,先收缩两条长边，再收缩两条短边
    :param xy_list:
    :param ratio:
    :return:
    """
    # 计算相邻两点的差  2-1 3-2 4-3 1-4
    delta_xy = np.zeros(shape=(4, 2), dtype=np.float)
    for i in range(4):
        delta_xy[i] = xy_list[(i + 1) % 4] - xy_list[i]
    # 计算欧式距离找长边  2-1和4-3 是否是长边,delta_xy[long_offset] 和delta_xy[(long_offset+2)]长
    dis = np.sum(np.square(delta_xy), axis=-1)
    if dis[0] + dis[2] >= dis[1] + dis[3]:
        # 先收缩长边再收缩短边
        xy_list[0] += delta_xy[0] * ratio
        xy_list[1] -= delta_xy[0] * ratio
        xy_list[2] += delta_xy[2] * ratio
        xy_list[3] -= delta_xy[2] * ratio
        # 重新计算间隔
        for i in range(4):
            delta_xy[i] = xy_list[(i + 1) % 4] - xy_list[i]
        # 收缩短边
        xy_list[1] += delta_xy[1] * ratio
        xy_list[2] -= delta_xy[1] * ratio
        xy_list[3] += delta_xy[3] * ratio
        xy_list[0] -= delta_xy[3] * ratio
        return xy_list
    # TODO 这里其实调换是有问题的,但是被机制的我解决了，调换回来就完事了
    else:  # 因为是顺时针的，所以调换一下边的顺序就可以复用
        tmp = xy_list[0]
        for index in range(3):
            xy_list[index] = xy_list[index + 1]
        xy_list[3] = tmp
        xy_list = shrink(xy_list, ratio)  # 代码复用
        tmp = xy_list[3]
        for index in range(3, 0, -1):
            xy_list[index] = xy_list[index - 1]
        xy_list[0] = tmp
        return xy_list


def point_inside_of_quad(px, py, quad_xy_list):
    """
    判断一个点是不是在四边形的内部 用向量叉积来算
    :param px:
    :param py:
    :param quad_xy_list:
    :return:
    """
    delta_xy_list = np.zeros((4, 2))
    delta_xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
    delta_xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
    yx_list = np.zeros((4, 2))
    yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
    a = delta_xy_list * ([py, px] - yx_list)
    b = a[:, 0] - a[:, 1]
    if np.amin(b) >= 0 or np.amax(b) <= 0:  # 判断是否同向
        return True
    else:
        return False


def create_gt_dir(info_file, original_gt_dir, gt_dir, scale_ratio_h, scale_ratio_w, pixel_size,
                  out_put_height, out_put_width, shrink_ratio):
    """
    创建缩小w,h之后，又缩小目标区域的ground truth
    步骤：1.加载原gt文件
          2.按顺时针进行排序
          3.按照  scale_ratio_h, scale_ratio_w 进行缩放
          4.进行shrink，使用shrink_ratio
          5.形成score map 和 geometry map 以及保存原坐标
          6.用npz格式存储于gt_dir
    :param info_file:
    :param original_gt_dir:
    :param gt_dir:
    :param scale_ratio_h:
    :param scale_ratio_w:
    :param pixel_size:
    :param out_put_height:
    :param out_put_width:
    :param shrink_ratio:
    :return:
    """
    with open(info_file, "r", encoding="utf-8") as info_file:
        lines = info_file.readlines()
        file_id_list = [line.strip().split(" ")[1] for line in lines]
        for file_id, _ in zip(file_id_list, tqdm(range(len(file_id_list)), desc="创建训练or验证or测试gt")):
            # 1.加载原gt文件
            original_gt_file_name = os.path.join(original_gt_dir, "gt_" + file_id + ".txt")
            # debug 用
            # print("processing：%s"%original_gt_file_name)
            # original_gt_file_name = r"H:\柳博的空间\data\ICDAR\robust_reading\Incidental_Scene_Text\ch4_training_localization_transcription_gt\gt_img_297.txt"
            with open(original_gt_file_name, "r", encoding="utf-8-sig") as f:  # 源文件有毒，用utf-8读取文件头有奇怪的符号
                lines = f.readlines()
                xy_list_array = np.zeros((len(lines), 4, 2), dtype=np.float)
                for i in range(len(lines)):
                    colunms = [int(num_str) for num_str in lines[i].strip().split(',')[:8]]
                    xy_list = np.reshape(np.array(colunms, dtype=np.float), (4, 2))
                    xy_list_array[i] = xy_list
                # 2.按顺时针进行排序
                for xy_list in xy_list_array:
                    xy_list = reorder_vertexes(xy_list)
                # 3.按照  scale_ratio_h, scale_ratio_w 进行缩放
                for xy_list in xy_list_array:
                    xy_list[:, 0] *= scale_ratio_w
                    xy_list[:, 1] *= scale_ratio_h
                original_coords = xy_list_array.copy()  # 保留一份原文件的拷贝  shape = (n,4,2) n 为框的数量
                # 4.进行shrink，使用shrink_ratio
                shrink_xy_list_array = np.zeros(shape=xy_list_array.shape)
                for i in range(len(xy_list_array)):
                    shrink_xy_list_array[i] = shrink(xy_list_array[i], shrink_ratio)
                # 5.形成score map 和 geometry map
                gt = np.zeros((out_put_height, out_put_width, 9),
                              dtype=np.float)  # gt[0]是score map ，gt[1:9]是geometry map
                for shrink_xy_list in shrink_xy_list_array:
                    input_min_xy = np.amin(shrink_xy_list, axis=0)
                    input_max_xy = np.amax(shrink_xy_list, axis=0)
                    output_min_xy = (input_min_xy / pixel_size).astype(int) - 2
                    output_max_xy = (input_max_xy / pixel_size).astype(int) + 2
                    # 保证不超过图像便捷
                    imin = np.maximum(0, output_min_xy[1])
                    imax = np.minimum(out_put_height, output_max_xy[1])
                    jmin = np.maximum(0, output_min_xy[0])
                    jmax = np.minimum(out_put_width, output_max_xy[0])
                    for i in range(imin, imax):
                        for j in range(jmin, jmax):
                            # TODO to be promoted
                            input_x = (j + 0.5) * config.pixel_size
                            input_y = (i + 0.5) * config.pixel_size
                            if point_inside_of_quad(input_x, input_y, shrink_xy_list):
                                gt[i, j, 0] = 1
                                # 剩下的8个通道是到四个点的坐标变换
                                for k in range(len(shrink_xy_list)):
                                    gt[i, j, 2 * k + 1] = input_x - shrink_xy_list[k][0]
                                    gt[i, j, 2 * k + 2] = input_y - shrink_xy_list[k][1]
                # 6.用npy格式存储于gt_dir
                gt_filename = os.path.join(gt_dir, file_id + "_gt.npz")
                np.savez(gt_filename, gt, original_coords)


# 第二步，生成训练集、验证集和测试集 包括 img、gt
def create_train_valid_test_img_gt():
    original_img_dirs = [config.original_train_img_dir, config.original_train_img_dir, config.original_test_img_dir]
    original_gt_dirs = [config.original_train_gt_dir, config.original_train_gt_dir, config.original_test_gt_dir]
    info_files = [config.train_info_file, config.valid_info_file, config.test_info_file]
    img_dirs = [config.train_img_dir, config.valid_img_dir, config.test_img_dir]
    gt_dirs = [config.train_gt_dir, config.valid_gt_dir, config.test_gt_dir]

    img_height = config.img_height
    img_width = config.img_width

    scale_ratio_h = config.scale_ratio_h
    scale_ratio_w = config.scale_ratio_w
    out_put_height = config.out_put_height
    out_put_width = config.out_put_width
    shrink_ratio = config.shrink_ratio
    pixel_size = config.pixel_size

    for dir in img_dirs + gt_dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
    for part_i in range(3):
        create_img_dir(info_files[part_i], original_img_dirs[part_i], img_dirs[part_i], img_height, img_width)
        create_gt_dir(info_files[part_i], original_gt_dirs[part_i], gt_dirs[part_i],
                      scale_ratio_h, scale_ratio_w, pixel_size, out_put_height, out_put_width, shrink_ratio)


if __name__ == "__main__":
    # create_info_file()
    create_train_valid_test_img_gt()
