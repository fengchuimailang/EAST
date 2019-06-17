import os

data_root = r"H:\柳博的空间\data\ICDAR\robust_reading\Incidental_Scene_Text"

# ICDAR 2013 robust_reading 数据集原文件
original_train_img_dir = os.path.join(data_root, "ch4_training_images")
original_test_img_dir = os.path.join(data_root, "ch4_test_images")
original_train_gt_dir = os.path.join(data_root, "ch4_training_localization_transcription_gt")
original_test_gt_dir = os.path.join(data_root, "Challenge4_Test_Task1_GT")

# 用于生成tfrecord的中间过程文件
validation_percent = 0.1
num_train_samples = 900
num_valid_samples = 100
num_test_samples = 500

img_height = 320
img_width = 640

scale_ratio_h = img_height / 720
scale_ratio_w = img_width / 1280

shrink_ratio = 0.2  # in paper it's 0.3, maybe to large to this problem
pixel_size = 4

out_put_height = int(img_height / pixel_size)
out_put_width = int(img_width / pixel_size)

max_original_coord_number = 200

train_info_file = os.path.join(data_root, "train.info")
valid_info_file = os.path.join(data_root, "valid.info")
test_info_file = os.path.join(data_root, "test.info")
train_img_dir = os.path.join(data_root, "train_img")
valid_img_dir = os.path.join(data_root, "valid_img")
test_img_dir = os.path.join(data_root, "test_img")
train_gt_dir = os.path.join(data_root, "train_gt")
valid_gt_dir = os.path.join(data_root, "valid_gt")
test_gt_dir = os.path.join(data_root, "test_gt")

show_gt_image_dir_name = os.path.join(data_root, "show_gt_images")
show_act_image_dir_name = os.path.join(data_root, "show_act_images")

shuffle_threads = 2

# 用于神经网络训练的处理好的文件
train_tfrecords = os.path.join(data_root, "train.tfrecords")
valid_tfrecords = os.path.join(data_root, "valid.tfrecords")
test_tfrecords = os.path.join(data_root, "test.tfrecords")

# 神经网络训练相关
logdir = "./logdir"
batch_size = 10
learning_rate = 1e-4
display = 10  # 每隔多少步进行输出
total_steps = 200000
view_valid_dir = os.path.join(data_root, "view_valid_dir")

# 神经网络测试相关
eval_times = 20

# 神经网络测试相关
view_test_dir = os.path.join(data_root, "view_test_dir")
