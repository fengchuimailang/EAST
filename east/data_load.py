import sys
sys.path.append("..")
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from PIL import Image
import east.config as config
import os
import time


def samples(img_dir, gt_dir, info_file):
    """
    创建训练、验证和测试的tfrecord 记录实例generator
    :param img_dir:
    :param gt_dir:
    :param info_file:
    :return:
    """
    with open(info_file, encoding="utf-8") as info_f:
        lines = [line.strip() for line in info_f.readlines()]
        for line in lines:
            img_id = line.split(" ")[1]
            # 打开图片
            image_filename = os.path.join(img_dir, img_id + "_input.jpg")
            image = Image.open(image_filename)
            image = np.array(image)
            # 打开ground_truth
            gt_filename = os.path.join(gt_dir, img_id + "_gt.npz")
            gt_and_original_coord = np.load(gt_filename)
            gt = gt_and_original_coord["arr_0"]
            original_coord = gt_and_original_coord["arr_1"]
            yield image, gt, original_coord, img_id


def store_tfrecords(img_dir, gt_dir, info_file, tfrecords_save_path, name):
    """
    创建tfrecords
    :param img_dir:
    :param gt_dir:
    :param info_file:
    :param tfrecords_save_path:
    :param name:
    :return:
    """
    count = 0
    writer = tf.python_io.TFRecordWriter(tfrecords_save_path)
    for image, ground_truth, original_coord, img_id in tqdm(samples(img_dir, gt_dir, info_file),
                                                            desc="创建%s的tfrecords" % name):
        # make features
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        _image_width = _int64_feature(image.shape[1])
        _image_height = _int64_feature(image.shape[0])
        _image = _bytes_feature(image.tobytes())
        _gt_width = _int64_feature(ground_truth.shape[1])
        _gt_height = _int64_feature(ground_truth.shape[0])
        _gt = _bytes_feature(ground_truth.tobytes())
        _original_coord_n = _int64_feature(original_coord.shape[0])
        _original_coord = _bytes_feature(original_coord.tobytes())
        _img_id = _bytes_feature(img_id.encode("utf-8"))  # 由string 转成字节形式
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'image_width': _image_width,
                'image_height': _image_height,
                'image': _image,
                'gt_width': _gt_width,
                'gt_height': _gt_height,
                "gt": _gt,
                "original_coord_n": _original_coord_n,
                "original_coord": _original_coord,
                "img_id": _img_id
            })
        )
        writer.write(example.SerializeToString())
        count += 1
    writer.close()
    print('%d samples generated.' % count)


def store():
    # store_tfrecords(config.train_img_dir, config.train_gt_dir, config.train_info_file, config.train_tfrecords, "训练")
    # store_tfrecords(config.valid_img_dir, config.valid_gt_dir, config.valid_info_file, config.valid_tfrecords, "验证")
    store_tfrecords(config.test_img_dir, config.test_gt_dir, config.test_info_file, config.test_tfrecords, "测试")


if __name__ == '__main__':
    start = time.time()
    store()
    print("Time: %f s." % (time.time() - start))
