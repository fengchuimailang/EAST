import sys
sys.path.append("..")
import tensorflow as tf
from east.east_net import EAST
import east.config as config
import time
import os
from PIL import Image, ImageDraw
from east.locality_aware_nms import nms_locality
import numpy as np


def turn_pred_delta_coord_to_coord(pred):
    """ 将一个预测的pred(score map +delta_coord)转化为(score map +coord)"""
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            input_y = (i + 0.5) * config.pixel_size
            input_x = (j + 0.5) * config.pixel_size
            for k in range(4):
                pred[i][j][2 * k + 1] = input_x - pred[i][j][2 * k + 1]
                pred[i][j][2 * k + 2] = input_y - pred[i][j][2 * k + 2]


def train():
    g = EAST(config=config)
    g.build_net()
    sv = tf.train.Supervisor(graph=g.graph, logdir=config.logdir)
    # merged_summary_op = tf.summary.merge_all()
    saver = sv.saver  # 创建saver
    with sv.managed_session() as sess:
        epoch = 0
        MLoss = 0
        S_M_loss = 0
        G_loss = 0
        time_start = time.time()
        valid_step = g.config.num_train_samples // g.config.batch_size  # 训练完一轮valid一次
        for step in range(1, g.config.total_steps):
            lr = g.config.learning_rate
            s_m_loss, g_loss, mloss, pred, original_coord, original_coord_n, img, score_map_and_coords, _ = \
                sess.run(
                    [g.s_m_loss, g.g_loss, g.mean_loss, g.pred, g.original_coord, g.original_coord_n, g.x, g.y,
                     g.optimizer],
                    {g.learning_rate: lr, g.train_stage: True})
            MLoss += mloss  # 平均每个batch_size的loss
            S_M_loss += s_m_loss
            G_loss += g_loss
            if step % g.config.display == 0:
                # 检验数据
                # for i in range(g.config.batch_size):
                #     img_i = (255 - img[i] * 255).astype(np.uint8)
                #     img_i = Image.fromarray(img_i)
                #     original_coord_i = original_coord[i]  # 第i张图片
                #     score_map_and_coords_i = score_map_and_coords[i]  # 第i个输出
                #     turn_pred_delta_coord_to_coord(score_map_and_coords_i)
                #     show_gt_img = img_i.copy()
                #     draw = ImageDraw.Draw(show_gt_img)
                # for j in range(original_coord_n[i]):  # 对于每一个ground_truth 进行验证
                #     draw.line(
                #         [tuple(original_coord_i[j][0]), tuple(original_coord_i[j][1]),
                #          tuple(original_coord_i[j][2]),
                #          tuple(original_coord_i[j][3]), tuple(original_coord_i[j][0])],
                #         width=2, fill='green')

                # for j in range(score_map_and_coords_i.shape[0]):    # 对于 每一个整理成tfrecord的score_map和coord 进行验证
                #     for k in range(score_map_and_coords_i.shape[1]):
                #         input_y = (j + 0.5) * config.pixel_size
                #         input_x = (k + 0.5) * config.pixel_size
                #         if score_map_and_coords_i[j][k][0] == 1:
                #             draw.point([(input_x, input_y)], fill="blue")
                #             coord_from_gt = score_map_and_coords_i[j][k][1:].reshape([4, 2])
                #             draw.line([tuple(coord_from_gt[0]), tuple(coord_from_gt[1]),
                #                        tuple(coord_from_gt[2]), tuple(coord_from_gt[3]),
                #                        tuple(coord_from_gt[0])], width=1, fill='blue')
                # show_gt_img.show()
                # input()

                print("step=%d,Loss=%f,S_M_loss=%f,G_loss=%f,time=%f" % (
                    step, MLoss / g.config.display, S_M_loss / g.config.display, G_loss / g.config.display,
                    time.time() - time_start))
                MLoss = 0
                S_M_loss = 0
                G_loss = 0

                # 保存模型
                # merged_summary = sess.run(merged_summary_op)
                # sv.summary_computed(sess, merged_summary, global_step=step)
                # saver.save(sess, config.logdir, global_step=step)

            if step % valid_step == 0:
                # if True:
                epoch += 1
                VLoss = 0
                count = g.config.num_valid_samples // g.config.batch_size
                for vi in range(count):
                    vloss, pred, original_coord, original_coord_n, img_id, img = sess.run(
                        [g.mean_loss, g.pred, g.original_coord, g.original_coord_n, g.img_id, g.x],
                        {g.learning_rate: lr, g.train_stage: False})
                    VLoss += vloss
                VLoss /= count  # 每个batch_size的loss

                # 在验证集上画出图像

                # 还原坐标
                for i in range(g.config.batch_size):
                    turn_pred_delta_coord_to_coord(pred[i])
                pred = pred.reshape(g.config.batch_size, -1, 9)
                for i in range(g.config.batch_size):
                    S = nms_locality(pred[i])
                    img_i = (255 - img[i] * 255).astype(np.uint8)
                    img_i = Image.fromarray(img_i)
                    show_gt_image = img_i.copy()
                    draw = ImageDraw.Draw(show_gt_image)
                    original_coord_i = original_coord[i]  # 第i张图片
                    img_id_i = img_id[i].decode()
                    for j in range(original_coord_n[i]):  # 对于每一个ground_truth
                        draw.line(
                            [tuple(original_coord_i[j][0]), tuple(original_coord_i[j][1]),
                             tuple(original_coord_i[j][2]),
                             tuple(original_coord_i[j][3]), tuple(original_coord_i[j][0])],
                            width=1, fill='green')
                    if (len(S) != 0):
                        S = S[:, 1:9].reshape([-1, 4, 2])
                        for j in range(S.shape[0]):
                            draw.line(
                                [tuple(S[j][0]), tuple(S[j][1]),
                                 tuple(S[j][2]),
                                 tuple(S[j][3]), tuple(S[j][0])],
                                width=1, fill='blue')
                    if not os.path.exists(g.config.view_valid_dir):
                        os.mkdir(g.config.view_valid_dir)
                    show_gt_image.save(os.path.join(g.config.view_valid_dir, img_id_i + "_view.jpg"))

                print("validation --- Loss=%f" % (VLoss))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
