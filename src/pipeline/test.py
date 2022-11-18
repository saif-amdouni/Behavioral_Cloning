import os
import sys
from PIL import Image
import cv2
from mss import mss
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator


import numpy as np

from time import time

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from shared.CONFIG import *
from shared.utils import countdown


from vjoy import vjoydevice
from model import create_VGG16base
from tensorflow.keras import optimizers

max_V_joy = 32768

j = vjoydevice.VJoyDevice(1)
j.set_axis(vjoydevice.HID_USAGE_Y, int(max_V_joy / 2))
j.set_axis(vjoydevice.HID_USAGE_X, int(max_V_joy / 2))


model = create_VGG16base()
checkpoint_path = os.path.join("../models/VGG16base", "cp-0005", "saved_model.pb")

checkpoint_dir = os.path.dirname(checkpoint_path)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5), metrics=['accuracy'])
print("model compiled")
print(f"loading from : {checkpoint_dir}")
model.load_weights(checkpoint_dir)
print("successfully loaded model")
countdown(5)
while 1:
    # start_time = time()
    screenshot = mss().grab(mon)  # get a screenshot of the screen
    img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))  # X
    img = np.expand_dims(img / 255, axis=0)
    prediction = model.predict(x=img)
    print(prediction)
    lateral = prediction[0][2] - prediction[0][1]
    forward = prediction[0][0]
    print(lateral)
    j.set_axis(vjoydevice.HID_USAGE_Y, int(max_V_joy * (-forward * 0.5 + 0.5)))
    j.set_axis(vjoydevice.HID_USAGE_X, int(max_V_joy * ((lateral + 1) / 2)))
    # print("FPS: ", 1.0 / (time() - start_time))
    # print(np.argmax(prediction))
