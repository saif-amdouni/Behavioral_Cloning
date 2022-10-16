"""
    file containing class and functions for collecting training data
"""

import os

import cv2
import numpy as np
from PIL import Image
from mss import mss
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
from shared.CONFIG import *
from getkeys import key_check
from shared.utils import countdown


def keys2output(keys):
    """
    Convert keys to a ...multi-hot... array
     0  1  2
    [Z, Q, D] boolean values.
    """
    if "Q" in ''.join(keys):
        return [1, 0, 0]
    elif "D" in ''.join(keys) :
        return [0, 0, 1]
    else:
        return [0, 1, 0]


class CollectTrainingData(object):

    def __init__(self):
        self.sct = mss()

    def collect(self):
        # collect images for training
        count = 0  # training packs count
        totalimg = 0
        countdown(Count_Down)
        print("Start collecting images...")
        try:
            # check if training file exists
            if not os.path.exists(training_dir):
                os.makedirs(training_dir)
            if os.path.isfile(training_dir + "/" + file_name):
                training_data = list(np.load(training_dir + "/" + file_name, allow_pickle=True))
            else:
                training_data = []

            while 1:
                screenshot = self.sct.grab(mon)  # get a screenshot of the screen
                img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (trainingImgWidth, trainingImgHeight))  # X

                keys = key_check()  # get user's input
                output = keys2output(keys)  # Y
                training_data.append([img, output])  # map img to input
                if len(training_data) % training_pack == 0:
                    count += 1
                    totalimg += 1
                    np.save(os.path.join(training_dir, f"{file_name}-{int(totalimg / 6)}"), training_data)
                    print(f"{totalimg * training_pack} images saved !")
                    if count == 6:
                        count = 0
                        training_data = []
        except Exception as e:
            print(e)

if __name__ == "__main__":
    collector = CollectTrainingData()
    collector.collect()