import tensorflow as tf


def total_loss(score_map_loss, geomery_loss, lambda_g=1):
    """
    总损失
    :param score_map_loss:
    :param geomery_loss:
    :param lambda_g:
    :return:
    """
    return score_map_loss + lambda_g * geomery_loss


def scope_map_loss(y_pred, y_true):
    # 类别平衡交叉熵
    # 这种方法不work，loss =nan ,更换成下面这种方法
    # 目的：balance positive and negative samples in an image
    # beta = 1 - tf.reduce_mean(y_true)
    # # first apply sigmoid activation
    # predicts = tf.nn.sigmoid(y_pred)
    # y_pred += 0.00001
    # class_balanced_cross_entropy = -1 * beta * y_true * tf.log(y_pred) - (1 - beta) * (1 - y_true) * tf.log(1 - y_pred)
    # return tf.reduce_mean(class_balanced_cross_entropy, axis=[1, 2, 3])

    # 第二种方法
    beta = (1 - tf.reduce_mean(y_true)) / tf.reduce_mean(y_true)
    loss_matrix = tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pos_weight=beta,
                                                           name="scope_map_loss")
    return tf.reduce_mean(loss_matrix)


def smooth_l1(x):
    x2 = tf.square(x)
    condition = tf.abs(x) > 1
    return tf.where(condition, tf.abs(x), x2)


def geomery_loss(y_pred, y_true, is_QUAD=True):
    if is_QUAD:
        box_diff = y_pred - y_true
        total_l1 = tf.reduce_sum(smooth_l1(box_diff))
        shape = tf.cast(tf.shape(y_pred), tf.float32)[1:3]
        return total_l1 / (tf.reduce_prod(shape) * 8.0)
    else:
        pass
