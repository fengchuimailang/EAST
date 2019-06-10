import os

for f in os.listdir(r"H:\柳博的空间\data\ICDAR\robust_reading\Incidental_Scene_Text\ch4_training_images"):
    gt_file = "gt_" + f.split(".")[0] + ".txt"
    with open(
            r"H:\柳博的空间\data\ICDAR\robust_reading\Incidental_Scene_Text\ch4_training_localization_transcription_gt\\" + gt_file,
            encoding="utf-8") as gt:
        for line in gt.readlines():
            print(line)
