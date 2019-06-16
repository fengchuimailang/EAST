import tensorflow as tf
from net import EAST
import config
import time
import os
from PIL import Image
import numpy as np


def train():
    g = EAST(config=config)
    g.build_net(g.config.train_tfrecords, g.config.valid_tfrecords)
    sv = tf.train.Supervisor(graph=g.graph, logdir=config.logdir)
    with sv.managed_session() as sess:
        epoch = 0
        best_loss = 1e8
        best_acc = -1e8
        not_improve_count = 0
        MLoss = 0
        S_M_loss = 0
        G_loss = 0
        last_valid_acc = 0
        last_train_acc = 0
        time_start = time.time()
        # st 是step
        for step in range(1, g.config.total_steps):
            lr = g.config.learning_rate
            s_m_loss, g_loss, mloss, _ = sess.run([g.s_m_loss, g.g_loss, g.mean_loss, g.optimizer],
                                                  {g.learning_rate: lr, g.train_stage: True})
            MLoss += mloss
            S_M_loss += s_m_loss
            G_loss += g_loss

            # # 调试信息
            # # print(x.shape)
            # score_map = y[0, :, :, 0]
            # print(score_map)
            # score_map_img = Image.fromarray(np.array(score_map).astype(np.uint8))
            # score_map_img.show()
            # img = Image.fromarray(np.array((1 - x[0]) * 255).astype(np.uint8))
            # img.show()
            # input()

            # 显示信息
            if step % g.config.display == 0:
                print("step=%d,Loss=%f,S_M_loss=%f,G_loss=%f,time=%f" % (
                    step, MLoss / g.config.display, S_M_loss / g.config.display, G_loss / g.config.display,
                    time.time() - time_start))
                MLoss = 0
                S_M_loss = 0
                G_loss =0
            # epoch validation
            valid_step = g.config.num_train_samples // g.config.batch_size
            if step % valid_step == 0:
                epoch += 1
                VLoss = 0
                count = g.config.num_valid_samples // g.config.batch_size
                for vi in range(count):
                    vloss = sess.run(g.mean_loss,
                                     {g.learning_rate: lr, g.train_stage: False})
                    VLoss += vloss
                VLoss /= count
                print("validation --- Loss=%f" % (
                    VLoss))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
