import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import config
from prepocess import shrink


# 判断一个点是不是在四边形的内部
def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        # xy_list 是上一个点减下一个点的坐标偏移
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]

        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False


# 定义QUAD 到四个顶点的坐标变换,在原图坐标中比较
def coord_shift(px, py, shrink_xy_list):
    res = []
    for x, y in shrink_xy_list:
        res.append(px - x)
        res.append(py - y)
    return res


def process_label(data_dir=config.data_root):
    with open(os.path.join(data_dir, config.valid_info_file), 'r') as f_val:
        f_list = f_val.readlines()
    with open(os.path.join(data_dir, config.train_info_file), 'r') as f_train:
        f_list.extend(f_train.readlines())
    for line, _ in zip(f_list, tqdm(range(len(f_list)))):
        line_cols = str(line).strip().split(' ')
        img_name = line_cols[1]
        # 宽和高都缩小为原来的1/4
        # TODO to be promoted
        height = config.img_height
        width = config.img_width
        # gt[0]是score_map gt[1:9]是坐标变换
        gt = np.zeros((height // config.pixel_size, width // config.pixel_size, 9), dtype=float)
        train_label_dir = os.path.join(config.data_root, config.train_label_dir)
        xy_list_array = np.load(os.path.join(train_label_dir,
                                             img_name + '.npy'))
        train_image_dir = os.path.join(data_dir, config.train_dir)

        for xy_list in xy_list_array:
            _, shrink_xy_list, _ = shrink(xy_list, config.shrink_ratio)
            # amin Return the minimum of an array or minimum along an axis
            p_min = np.amin(shrink_xy_list, axis=0)
            p_max = np.amax(shrink_xy_list, axis=0)
            # TODO to be promoted
            # floor of the float
            ji_min = (p_min / config.pixel_size - 0.5).astype(int) - 1
            # +1 for ceil of the float and +1 for include the end
            # 外接矩形
            ji_max = (p_max / config.pixel_size - 0.5).astype(int) + 3
            imin = np.maximum(0, ji_min[1])
            imax = np.minimum(height // config.pixel_size, ji_max[1])
            jmin = np.maximum(0, ji_min[0])
            jmax = np.minimum(width // config.pixel_size, ji_max[0])
            for i in range(imin, imax):
                for j in range(jmin, jmax):
                    # 还原到原图中
                    px = (j + 0.5) * config.pixel_size
                    py = (i + 0.5) * config.pixel_size
                    if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):
                        gt[i, j, 0] = 1
                        # 剩下的8个通道是到四个点的坐标变换
                        shift = coord_shift(px, py, shrink_xy_list)
                        for k in range(1, 9):
                            gt[i, j, k] = shift[k - 1]

        np.save(os.path.join(train_label_dir, img_name + "_gt.npy"), gt)


if __name__ == "__main__":
    process_label()
