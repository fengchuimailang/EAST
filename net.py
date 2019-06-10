import tensorflow as tf
from loss import total_loss, scope_map_loss, geomery_loss


class EAST(object):
    def __init__(self, config):
        self.graph = tf.Graph()
        self.config = config

    def unpool(self, x, name="unpool"):
        """
        https://github.com/tensorflow/tensorflow/issues/2169
        N-dimensional version of the unpooling operation

        :param x: A Tensor of shape [b, d0, d1, ..., dn, ch]
        :param name:
        :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
        """
        with tf.name_scope(name) as scope:
            out = tf.concat([x, tf.zeros_like(x)], 3)
            out = tf.concat([out, tf.zeros_like(out)], 2)

            sh = x.get_shape().as_list()
            if None not in sh[1:]:
                out_size = [sh[0], sh[1] * 2, sh[2] * 2, sh[3]]
                out = tf.reshape(out, out_size)
            else:
                shv = tf.shape(x)
                ret = tf.reshape(out, tf.stack([sh[0], shv[1] * 2, shv[2] * 2, sh[3]]))
                out = ret
        return out

    def build_PVANET_based_net(self, batch_size, img_width, img_length, inputs):
        # feature extractor stem
        conv0 = tf.layers.conv2d(inputs, filters=16, kernel_size=(7, 7), strides=(2, 2), padding="same",
                                 activation=tf.nn.relu, name="conv0")

        conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                 activation=tf.nn.relu, name="conv0")

        # feature-merging branch

        # output layer

    def build_VFF16_based_net(self, inputs):
        # feature extractor stem
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

        # feature-merging branch

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

        score_map = tf.layers.conv2d(final_feature_map, filters=1, kernel_size=(1, 1), strides=(1, 1),
                                     padding="same",
                                     activation=tf.nn.relu, name="score_map")
        QUAD_coord = tf.layers.conv2d(final_feature_map, filters=8, kernel_size=(1, 1), strides=(1, 1),
                                      padding="same",
                                      activation=tf.nn.relu, name="QUAD_coord")

        # 目标输出 第一层scope_map 后面8层是坐标
        # targets = tf.placeholder(tf.int32, shape=[batch_size, img_width, img_height, 9], name="targets")

        # # loss
        # s_m_loss = scope_map_loss(score_map, targets[0])
        # g_loss = geomery_loss(QUAD_coord, targets[1:])
        # loss = total_loss(s_m_loss, g_loss)

        # 优化器
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

        # 返回结果
        return score_map, QUAD_coord

    def read_and_decode(self, save_paths, is_training=True):
        filename_queue = tf.train.string_input_producer([save_paths])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        context_features = {
            "image_width": tf.FixedLenFeature([], dtype=tf.int64),
            "image_height": tf.FixedLenFeature([], dtype=tf.int64),
            "image": tf.FixedLenFeature([], dtype=tf.string),
            # "coords": tf.FixedLenFeature([], dtype=tf.string),
            "gt_width": tf.FixedLenFeature([], dtype=tf.int64),
            'gt_height': tf.FixedLenFeature([], dtype=tf.int64),
            "gt": tf.FixedLenFeature([], dtype=tf.string)
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
        # coords = tf.decode_raw(context_parsed["coords"], tf.float32)
        gt_width = tf.cast(context_parsed["gt_width"], tf.int32)
        gt_height = tf.cast(context_parsed["gt_height"], tf.int32)
        gt = tf.decode_raw(context_parsed["gt"], tf.float64)

        image = tf.reshape(image, [image_height, image_width, 3])
        image = tf.cast(image, dtype=tf.float32) / 255.0
        # coords = tf.reshape(coords, [-1, 4, 2])
        gt = tf.reshape(gt, [gt_height, gt_width, 9])
        gt = tf.cast(gt, dtype=tf.float32)
        # input_tensors = [image, coords]
        input_tensors = [image, gt]

        shuffle_queue = tf.RandomShuffleQueue(self.config.batch_size * 100, self.config.batch_size * 10,
                                              dtypes=[t.dtype for t in input_tensors])
        enqueue_op = shuffle_queue.enqueue(input_tensors)
        runner = tf.train.QueueRunner(shuffle_queue, [enqueue_op] * self.config.shuffle_threads)
        tf.train.add_queue_runner(runner)
        output_tensors = shuffle_queue.dequeue()
        for i in range(len(input_tensors)):
            output_tensors[i].set_shape(input_tensors[i].shape)
        return tf.train.batch(tensors=output_tensors, batch_size=self.config.batch_size, dynamic_pad=True)

    def build_net(self, train_tfrecords, valid_tfrecords, is_training=True):
        with self.graph.as_default():
            if is_training:
                self.train_stage = tf.placeholder(tf.bool, shape=())  # True if train, else valid
                train_x, train_y = self.read_and_decode(train_tfrecords)
                valid_x, valid_y = self.read_and_decode(valid_tfrecords)
                self.x = tf.cond(self.train_stage, lambda: train_x, lambda: valid_x)
                self.y = tf.cond(self.train_stage, lambda: train_y, lambda: valid_y)
            else:
                self.x = tf.placeholder(tf.float32, shape=(None, self.config.img_height, self.config.img_width, 3))
            score_map, QUAD_coord = self.build_VFF16_based_net(self.x)

            # loss  y的第一层是score_map 后面8层是geomery信息
            s_m_loss = scope_map_loss(score_map, self.y[:, :, :, :1])
            g_loss = geomery_loss(QUAD_coord, self.y[:, :, :, 1:])
            loss = total_loss(s_m_loss, g_loss)
            self.mean_loss = tf.reduce_mean(loss)
            self.learning_rate = tf.placeholder(tf.float32, [])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-9)
            self.optimizer = optimizer.minimize(self.mean_loss)

            ## accuracy  TODO


if __name__ == "__main__":
    pass
