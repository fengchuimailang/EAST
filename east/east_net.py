import sys
sys.path.append("..")
import sys
sys.path.append("..")
import tensorflow as tf
from east.loss import total_loss, scope_map_loss, geomery_loss


class EAST(object):
    def __init__(self, config):
        self.graph = tf.Graph()
        self.config = config

    def load_tfrecords(self, save_paths, is_training=True):
        """
        读取输入并转换为dataset
        :param save_paths:
        :param is_training:
        :return:
        """

        def parse_example(serialized_example):
            context_features = {
                "image_width": tf.FixedLenFeature([], dtype=tf.int64),
                "image_height": tf.FixedLenFeature([], dtype=tf.int64),
                "image": tf.FixedLenFeature([], dtype=tf.string),
                "gt_width": tf.FixedLenFeature([], dtype=tf.int64),
                'gt_height': tf.FixedLenFeature([], dtype=tf.int64),
                "gt": tf.FixedLenFeature([], dtype=tf.string),
                "original_coord_n": tf.FixedLenFeature([], dtype=tf.int64),
                "original_coord": tf.FixedLenFeature([], dtype=tf.string),
                "img_id": tf.FixedLenFeature([], dtype=tf.string)
            }
            sequence_features = {
            }
            context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized_example,
                context_features=context_features,
                sequence_features=sequence_features
            )
            image_width = tf.cast(context_parsed["image_width"], tf.int32)
            image_height = tf.cast(context_parsed["image_height"], tf.int32)
            image = tf.decode_raw(context_parsed["image"], tf.uint8)
            # 图片预处理
            image = tf.reshape(image, [image_height, image_width, 3])
            image = tf.cast(image, dtype=tf.float32) / 255.0

            gt_width = tf.cast(context_parsed["gt_width"], tf.int32)
            gt_height = tf.cast(context_parsed["gt_height"], tf.int32)
            gt = tf.decode_raw(context_parsed["gt"], tf.float64)
            gt = tf.reshape(gt, [gt_height, gt_width, 9])
            gt = tf.cast(gt, dtype=tf.float32)

            original_coord_n = tf.cast(context_parsed["original_coord_n"], tf.int32)
            original_coord = tf.decode_raw(context_parsed["original_coord"], tf.float64)
            original_coord = tf.reshape(original_coord, [original_coord_n, 4, 2])

            img_id = tf.cast(context_parsed["img_id"], tf.string)  # 暂时没有用
            return image, gt, original_coord, original_coord_n, img_id

        dataset = tf.data.TFRecordDataset(save_paths)
        dataset = dataset.map(parse_example)
        dataset = dataset.repeat().shuffle(10 * self.config.batch_size)
        dataset = dataset.padded_batch(self.config.batch_size, ([self.config.img_height, self.config.img_width, 3],
                                                                [self.config.out_put_height, self.config.out_put_width,
                                                                 9], [self.config.max_original_coord_number, 4, 2], [],
                                                                []))
        iterator = dataset.make_one_shot_iterator()
        image, gt, original_coord, original_coord_n, img_id = iterator.get_next()
        return image, gt, original_coord, original_coord_n, img_id

    # def unpool(self, x, name="unpool"):
    #     """
    #     反池化
    #     https://github.com/tensorflow/tensorflow/issues/2169
    #     N-dimensional version of the unpooling operation
    #     :param x: A Tensor of shape [b, d0, d1, ..., dn, ch]
    #     :param name:
    #     :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    #     """
    #     with tf.name_scope(name) as scope:
    #         out = tf.concat([x, tf.zeros_like(x)], 3)
    #         out = tf.concat([out, tf.zeros_like(out)], 2)
    #
    #         sh = x.get_shape().as_list()
    #         if None not in sh[1:]:
    #             out_size = [sh[0], sh[1] * 2, sh[2] * 2, sh[3]]
    #             out = tf.reshape(out, out_size)
    #         else:
    #             shv = tf.shape(x)
    #             ret = tf.reshape(out, tf.stack([sh[0], shv[1] * 2, shv[2] * 2, sh[3]]))
    #             out = ret
    #     return out

    def unpool(self, value, name='unpool'):
        """N-dimensional version of the unpooling operation from
        https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

        :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
        :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
        """
        with tf.name_scope(name) as scope:
            sh = value.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(value, [-1] + sh[-dim:]))
            for i in range(dim, 0, -1):
                out = tf.concat([out, tf.zeros_like(out)], i)
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
            out = tf.reshape(out, out_size, name=scope)
        return out

    def build_PVANET_based_net(self, inputs):
        """
        构建以PVANET为backbone的神经网络
        :param inputs:
        :return:
        """
        # TODO
        pass

    def build_VFF16_based_net(self, inputs):
        """
        构建以VFF16为backbone的神经网络
        :param inputs:
        :return:
        """
        # feature extractor stem 特征提取
        conv1_1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv1_1")
        conv1_2 = tf.layers.conv2d(conv1_1, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv1_2")
        pool_1 = tf.layers.max_pooling2d(conv1_2, pool_size=[2, 2], strides=[2, 2], padding="same", name="pool_1")

        conv2_1 = tf.layers.conv2d(pool_1, filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv2_1")
        conv2_2 = tf.layers.conv2d(conv2_1, filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv2_2")
        pool_2 = tf.layers.max_pooling2d(conv2_2, pool_size=[2, 2], strides=[2, 2], padding="same", name="pool_2")

        conv3_1 = tf.layers.conv2d(pool_2, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv3_1")
        conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv3_2")
        conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv3_3")
        pool_3 = tf.layers.max_pooling2d(conv3_3, pool_size=[2, 2], strides=[2, 2], padding="same", name="pool_3")

        conv4_1 = tf.layers.conv2d(pool_3, filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv4_1")
        conv4_2 = tf.layers.conv2d(conv4_1, filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv4_2")
        conv4_3 = tf.layers.conv2d(conv4_2, filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv4_3")
        pool_4 = tf.layers.max_pooling2d(conv4_3, pool_size=[2, 2], strides=[2, 2], padding="same", name="pool_4")

        conv5_1 = tf.layers.conv2d(pool_4, filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv5_1")
        conv5_2 = tf.layers.conv2d(conv5_1, filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv5_2")
        conv5_3 = tf.layers.conv2d(conv5_2, filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   activation=tf.nn.relu, name="conv5_3")
        pool_5 = tf.layers.max_pooling2d(conv5_3, pool_size=[2, 2], strides=[2, 2], padding="same", name="pool_5")

        # feature-merging branch 特征融合

        unpool_1 = self.unpool(pool_5, "unpool_1")
        concat_1 = tf.concat([unpool_1, pool_4], axis=-1)
        merging_cov1_1 = tf.layers.conv2d(concat_1, filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                          activation=tf.nn.relu, name="merging_cov1_1")
        merging_cov1_2 = tf.layers.conv2d(merging_cov1_1, filters=512, kernel_size=(3, 3), strides=(1, 1),
                                          padding="same",
                                          activation=tf.nn.relu, name="merging_cov1_2")

        unpool_2 = self.unpool(merging_cov1_2, "unpool_2")
        concat_2 = tf.concat([unpool_2, pool_3], axis=-1)
        merging_cov2_1 = tf.layers.conv2d(concat_2, filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                          activation=tf.nn.relu, name="merging_cov2_1")
        merging_cov2_2 = tf.layers.conv2d(merging_cov2_1, filters=256, kernel_size=(3, 3), strides=(1, 1),
                                          padding="same",
                                          activation=tf.nn.relu, name="merging_cov2_2")

        unpool_3 = self.unpool(merging_cov2_2, "unpool_3")
        concat_3 = tf.concat([unpool_3, pool_2], axis=-1)
        merging_cov3_1 = tf.layers.conv2d(concat_3, filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                          activation=tf.nn.relu, name="merging_cov3_1")
        merging_cov3_2 = tf.layers.conv2d(merging_cov3_1, filters=128, kernel_size=(3, 3), strides=(1, 1),
                                          padding="same",
                                          activation=tf.nn.relu, name="merging_cov3_2")

        final_feature_map = tf.layers.conv2d(merging_cov3_2, filters=32, kernel_size=(3, 3), strides=(1, 1),
                                             padding="same",
                                             activation=tf.nn.relu, name="final_feature_map")

        # output layer  输出层
        score_map_losits = tf.layers.conv2d(final_feature_map, filters=1, kernel_size=(1, 1), strides=(1, 1),
                                            padding="same",
                                            activation=tf.nn.relu, name="score_map")
        score_map = tf.nn.sigmoid(score_map_losits, name="score_map_sigmoid")
        QUAD_coord = tf.layers.conv2d(final_feature_map, filters=8, kernel_size=(1, 1), strides=(1, 1),
                                      padding="same",
                                      activation=tf.nn.relu, name="QUAD_coord")
        return score_map, QUAD_coord

    def build_net(self, is_training=True):
        """
        构建神经网络包括输入和输出，再训练和测试的情况下都可以使用
        :param train_tfrecords:
        :param valid_tfrecords:
        :param is_training:
        :return:
        """
        with self.graph.as_default():
            if is_training:  # 训练中 包含训练阶段和测试阶段
                self.train_stage = tf.placeholder(tf.bool, shape=())  # True if train, else valid
                train_x, train_y, train_original_coord, train_original_coord_n, train_img_id = self.load_tfrecords(
                    self.config.train_tfrecords)
                valid_x, valid_y, valid_original_coord, valid_original_coord_n, valid_img_id = self.load_tfrecords(
                    self.config.valid_tfrecords)
                self.x = tf.cond(self.train_stage, lambda: train_x, lambda: valid_x)
                self.y = tf.cond(self.train_stage, lambda: train_y, lambda: valid_y)
                self.original_coord = tf.cond(self.train_stage, lambda: train_original_coord,
                                              lambda: valid_original_coord)
                self.original_coord_n = tf.cond(self.train_stage, lambda: train_original_coord_n,
                                                lambda: valid_original_coord_n)

                self.img_id = tf.cond(self.train_stage, lambda: train_img_id, lambda: valid_img_id)
            else:  # 运行某一个sample
                self.x = tf.placeholder(tf.float32, shape=(None, self.config.img_height, self.config.img_width, 3))
            score_map, QUAD_coord = self.build_VFF16_based_net(self.x)
            self.pred = tf.concat([QUAD_coord, score_map], -1)  # TODO 这里为了和nms对应,颠倒了顺序

            if is_training:
                # loss  y的第一层是score_map 后面8层是geomery信息
                self.s_m_loss = scope_map_loss(score_map, self.y[:, :, :, :1])
                self.g_loss = geomery_loss(QUAD_coord, self.y[:, :, :, 1:])
                loss = total_loss(self.s_m_loss, self.g_loss)
                self.mean_loss = loss
                self.learning_rate = tf.placeholder(tf.float32, [])
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-9)
                self.optimizer = optimizer.minimize(self.mean_loss)
