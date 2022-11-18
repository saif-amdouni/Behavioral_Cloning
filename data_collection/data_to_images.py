import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
from shared.CONFIG import data_dir, training_dir, csv

training_files = []
columns = ["ID", "path", "target"]

row_list = []

for entry in os.listdir(os.path.join(data_dir,training_dir)):
    if os.path.isfile(os.path.join(data_dir,training_dir, entry)):
        training_files.append(entry)

for j, file_path in enumerate(training_files):

    os.mkdir(os.path.join(data_dir,training_dir, file_path[:-4]))
    os.mkdir(os.path.join(data_dir,training_dir, file_path[:-4], "left"))
    os.mkdir(os.path.join(data_dir,training_dir, file_path[:-4], "right"))
    os.mkdir(os.path.join(data_dir,training_dir, file_path[:-4], "forward"))

    data = np.load(os.path.join(data_dir, training_dir, file_path), allow_pickle=True)
    # print(data.shape[0])

    for i, pic in tqdm(enumerate(data),total=len(data)):
        target = -1
        img_path = ""
        if pic[1][0] == 1:
            img_path = os.path.join(data_dir, training_dir, file_path[:-4], "left", f"package{j}_image{i}.jpg")
            cv2.imwrite(img_path, pic[0])
            target = 0
        elif pic[1][1] == 1:
            img_path = os.path.join(data_dir, training_dir, file_path[:-4], "forward", f"package{j}_image{i}.jpg")
            cv2.imwrite(img_path, pic[0])
            target = 1
        elif pic[1][2] == 1:
            img_path = os.path.join(data_dir, training_dir, file_path[:-4], "right", f"package{j}_image{i}.jpg")
            cv2.imwrite(img_path, pic[0])
            target = 2
        row_list.append({"ID": int(str(j)+str(i)), "path": img_path, "target": target})
        # print(100 * i / data.shape[0])
train_csv = pd.DataFrame(row_list, columns=columns)
train_csv.to_csv(os.path.join(data_dir, csv), index=False)
