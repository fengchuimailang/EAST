import tensorflow as tf
import os
import time
from PIL import Image, ImageDraw
import numpy as np

import east.config as config
from east.east_net import EAST
from east.train import turn_pred_delta_coord_to_coord
from east.locality_aware_nms import nms_locality


def eval():
    g = EAST(config)
    g.build_net(is_training=False)
    print("Graph loaded.")
    with g.graph.as_default():
        test_x, test_y, test_original_coord, test_original_coord_n, test_img_id = g.load_tfrecords(
            g.config.test_tfrecords)
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(config.logdir))
            print(tf.train.latest_checkpoint(config.logdir))
            print("Restored!")

            for k in range(config.eval_times):
                x, y, original_coord, original_coord_n, img_id, = sess.run(
                    [test_x, test_y, test_original_coord, test_original_coord_n, test_img_id])

                pred = sess.run(g.pred, {g.x: x})
                # 还原坐标
                for i in range(g.config.batch_size):
                    turn_pred_delta_coord_to_coord(pred[i])
                pred = pred.reshape(g.config.batch_size, -1, 9)
                for i in range(g.config.batch_size):
                    S = nms_locality(pred[i])
                    img_i = (255 - x[i] * 255).astype(np.uint8)
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
                    if not os.path.exists(g.config.view_test_dir):
                        os.mkdir(g.config.view_test_dir)
                    show_gt_image.show()
                    show_gt_image.save(os.path.join(g.config.view_test_dir, img_id_i + "_view.jpg"))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    eval()
