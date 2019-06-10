from __future__ import print_function
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import collections
import random
import time
import io

import config


# 分割训练街和测试集id 到两个不同的文件中
def split_train_valid_to_info(train_valid_dir, train_info_file, valid_info_file, ratio=0.1):
    with io.open(train_info_file, 'w', encoding='utf-8') as t_f:
        with io.open(valid_info_file, 'w', encoding='utf-8') as v_f:
            train_count = 0
            valid_count = 0
            for filename in tqdm(os.listdir(train_valid_dir)):
                if valid_count < ratio * (config.num_train_samples + config.num_valid_samples):
                    valid_count += 1
                    v_f.write("valid_number:" + str(valid_count) + " " + filename.split(".")[0] + "\n")
                else:
                    train_count += 1
                    t_f.write("train_number:" + str(train_count) + " " + filename.split(".")[0] + "\n")


# def test_to_info(test_dir, test_info_file):
#     with open(test_info_file, "w", encoding="utf-8") as t_f:
#         test_count = 0
#         for filename in tqdm(os.listdir(test_dir)):
#             test_count += 1
#             t_f.write("test_number:" + str(test_count) + " " + filename.split(".")[0] + "\n")


def image_path_from(jpg_root_dir, img_id):
    filename = os.path.join(jpg_root_dir, img_id + ".jpg")
    if os.path.exists(filename):
        return filename
    else:
        raise Exception("no image file found by img_id：%s" % img_id)


def label_path_from(label_root_dir, img_id):
    filename = os.path.join(label_root_dir, "gt_" + img_id + ".txt")
    if os.path.exists(filename):
        return filename
    else:
        raise Exception("no label txt file found by img_id：%s" % img_id)


# 读取图片和ground_truth,这里图片不缩放
def samples(jpg_root_dir, label_dir, info_file):
    with open(info_file, encoding="utf-8") as info_f:
        line = info_f.readline().strip()
        while line:
            info_id, img_id = line.split(" ")
            # 打开图片
            image = Image.open(image_path_from(jpg_root_dir, img_id))
            # 打开groundtruth
            coords = []
            texts = []
            with open(label_path_from(label_dir, img_id), encoding="utf-8") as label_f:
                for label_line in label_f.readlines():
                    label_line = label_line.strip()
                    if not label_line:
                        strs = label_line.split(",")
                        coord_strs = strs[:-1]
                        assert len(coord_strs) == 8, "坐标长不为8"
                        text = strs[-1]
                        coords.append([int(i) for i in coord_strs])
                        texts.append(text)
            line = info_f.readline().strip()

            # 加载ground_truth
            ground_truth = np.load(os.path.join(config.data_root, config.train_label_dir, img_id + "_gt.npy"))
            # yield image, coords
            yield image, ground_truth


# def summary():
#     count = 0
#     exceed_count = 0
#     max_width = 0
#     max_label_length = 0
#     width_dict = collections.defaultdict(int)
#     for image, label in samples(info_file):
#         count += 1
#         width, height = image.size
#         if height != config.max_height:
#             new_width = float(width) * float(config.max_height) / float(height)
#             new_width = int(new_width)
#             width_dict[new_width] += 1
#             if new_width > config.max_width:
#                 exceed_count += 1
#                 print('%d Exceed max width.' % exceed_count)
#                 continue
#             if new_width > max_width:
#                 max_width = new_width
#         if len(label) > max_label_length:
#             max_label_length = len(label)
#     print("total images: %d, max width: %d, max label length: %d" % (count, max_width, max_label_length))
#     sorted_width = [(k, width_dict[k]) for k in sorted(width_dict.keys())]
#     tmp = 0.0
#     for k, v in sorted_width:
#         tmp += float(v)
#         print(k, v, tmp / float(count))


def shuffle_lines(filename, save_path):
    with io.open(filename, 'r', encoding='utf-8') as f:
        raw = f.readlines()
    random.shuffle(raw)
    with io.open(save_path, 'w', encoding='utf-8') as g:
        for line in raw:
            g.write(line)


def store_tfrecords(jpg_root_dir, label_dir, info_file, save_path):
    count = 0
    writer = tf.python_io.TFRecordWriter(save_path)
    for image, ground_truth in tqdm(samples(jpg_root_dir, label_dir, info_file)):
        # TODO 图片大小缩放
        # # resize image to max height
        # width, height = image.size
        # if height > config.final_height:
        #     new_width = int(float(width) * float(config.final_height) / float(height))
        #     if new_width > config.max_width or new_width <= 0:
        #         continue
        image = image.resize((config.img_width, config.img_height), Image.NEAREST).convert('RGB')
        image = np.array(image)
        ## TODO 图片padding
        # left_pad = int(0.01 * config.max_width)
        # top_pad = int(random.random() * (config.final_height - image.shape[0]))
        # image = np.pad(image, ((top_pad, 0), (left_pad, 0)), 'constant', constant_values=(255, 255))
        # if image.shape[1] > config.max_width or image.shape[0] > config.final_height:
        #     continue

        # 图片翻转
        image = 255 - image

        # transform label string to int array

        # coords = np.array(coords, np.int32)
        # coords_len = len(coords)
        # make features
        _image_width = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]]))
        _image_height = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]]))
        # _coords_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[coords_len]))
        _image = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        # _coords = tf.train.Feature(bytes_list=tf.train.BytesList(value=[coords.tobytes()]))
        _gt_width = tf.train.Feature(int64_list=tf.train.Int64List(value=[ground_truth.shape[1]]))
        _gt_height = tf.train.Feature(int64_list=tf.train.Int64List(value=[ground_truth.shape[0]]))
        _gt = tf.train.Feature(bytes_list=tf.train.BytesList(value=[ground_truth.tobytes()]))

        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'image_width': _image_width,
                'image_height': _image_height,
                # 'label_length': _coords_length,
                'image': _image,
                # "coords": _coords
                'gt_width': _gt_width,
                'gt_height': _gt_height,
                "gt": _gt
            })
        )
        writer.write(example.SerializeToString())
        count += 1
    writer.close()
    print('%d samples generated.' % count)


# def test():
#     # split_train_valid_to_info(config.train_dir, config.train_info_file, config.valid_info_file)
#     test_to_info(config.test_dir, config.test_info_file)
#     # shuffle_lines(config.train_info_file, config.train_info_file)
#     # shuffle_lines(config.valid_info_file, config.valid_info_file)
#     # shuffle_lines(config.test_info_file, config.test_info_file)


def store():
    store_tfrecords(config.train_dir, config.train_gt_dir, config.train_info_file, config.train_tfrecords)
    store_tfrecords(config.train_dir, config.train_gt_dir, config.valid_info_file, config.valid_tfrecords)
    store_tfrecords(config.test_dir, config.test_gt_dir, config.test_info_file, config.test_tfrecords)


def load_tfrecords(tfrecord_path):
    def parse_example(serialized_example):
        context_features = {
            "image_width": tf.FixedLenFeature([], dtype=tf.int64),
            "image_height": tf.FixedLenFeature([], dtype=tf.int64),
            "image": tf.FixedLenFeature([], dtype=tf.string),
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
        gt = tf.reshape(gt, [gt_height, gt_width, 9])
        gt = tf.cast(gt, dtype=tf.float32)
        # input_tensors = [image, coords]
        return image_width,image_height,image, gt

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)
    dataset = dataset.repeat().shuffle(10 * config.batch_size)
    # 每一条数据长度不一致时，用padded_batch进行补全操作
    # dataset = dataset.padded_batch(config.batch_size, ([config.image_height, config.image_max_width, 1],
    #                                                    [config.label_max_len], [lfv], [lfv], [lfv, 4]))
    # make_one_shot_iterator: Creates an `Iterator` for enumerating the elements of this dataset.
    iterator = dataset.make_one_shot_iterator()
    image_width,image_height,image, gt = iterator.get_next()
    return image_width,image_height,image, gt

def loaded_correctly():
    image_width,image_height,image, gt = load_tfrecords(config.train_tfrecords)

    with tf.Session() as sess:
        width,height,img = sess.run([image_width,image_height,image])
        print(repr(img))

if __name__ == '__main__':
    start = time.time()
    # test()
    store()
    # raw_to_info(config.test_info_file_raw, config.test_info_file)
    # save_alphabet(config.train_info_file_raw, config.test_info_file_raw, config.alphabet_path)
    print("Time: %f s." % (time.time() - start))
    # loaded_correctly()


