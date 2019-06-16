data_root = r"H:\柳博的空间\data\ICDAR\robust_reading\Incidental_Scene_Text"
train_dir = data_root + r"\ch4_training_images"
test_dir = data_root + r"\ch4_test_images"
train_gt_dir = data_root + r"\ch4_training_localization_transcription_gt"
test_gt_dir = data_root + r"\Challenge4_Test_Task1_GT"

batch_size = 5
validation_percent = 0.1

num_train_samples = 900
num_valid_samples = 100
num_test_samples = 500

train_info_file = data_root + r'\train.info'
valid_info_file = data_root + r'\valid.info'
test_info_file = data_root + r'\test.info'
train_tfrecords = data_root + r'\train.tfrecords'
valid_tfrecords = data_root + r'\valid.tfrecords'
test_tfrecords = data_root + r'\test.tfrecords'

check_point = data_root + r"\checkpoint"
# img_height = int(720 / 2)
# img_width = int(1280 / 2)
img_height = 320
img_width = 640


learning_rate = 1e-4

logdir = data_root + r"\logdir"
shuffle_threads = 2

show_gt_image_dir_name = 'show_gt_images'
show_act_image_dir_name = 'show_act_images'
train_label_dir = 'labels'

# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2

pixel_size = 4
# 每隔多少步进行输出
display = 10

total_steps = 200000
