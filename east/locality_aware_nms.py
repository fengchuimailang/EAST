import sys
sys.path.append("..")
import numpy as np
from shapely.geometry import Polygon
import os


def intersection(g, p):
    # 创建带比较的两个点的多边形
    g_poly = Polygon(g[1:9].reshape((4, 2))).convex_hull
    p_ploy = Polygon(p[1:9].reshape((4, 2))).convex_hull
    inter = g_poly.intersection(p_ploy).area
    union = g_poly.area + g_poly.area - inter
    if union == 0:
        return 0
    else:
        return inter / union  # 交并比


def weighted_merge(g, p):  # 它这里的第9层是score_map  把p合并到q上
    g[1:9] = (g[0] * g[1:9] + p[0] * p[1:9]) / (g[0] + p[0])  # 坐标取加权平均
    g[0] = (g[0] + p[0])  # 分数进行累加
    return g


def standard_nms(S, iou_thres):
    order = np.argsort(S[:, 0])[::-1]  # argsort 返回的是index
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])  # 对于剩下的每一个多边形，计算IOU

        inds = np.where(ovr <= iou_thres)[0]  # 保留IOU小于阈值的index
        order = order[inds + 1]  # 把index向前移一位，因为order[0]已经被keep了

    return S[keep]  # 惊讶于numpy竟然可以做这种骚操作！


def nms_locality(polys, confidence_thres=0.4, iou_thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first prob, then 8 coordinates
    :return: boxes after nms
    '''
    confidence_thres_condition = np.where(polys[:, 0] > confidence_thres)
    polys = polys[confidence_thres_condition]
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > iou_thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), iou_thres)


if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    # print(Polygon(np.array([[0, 0], [50, 0],
    #                         [50, 50], [0, 50]])).area)
    test = np.load(r"H:\柳博的空间\EAST\east\test_nms.npy")
    S = standard_nms(test, iou_thres=0.3)
    print(S)
