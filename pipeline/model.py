import sys
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from shared.CONFIG import *

def create_VGG16base(summary = False):
    conv_base = VGG16(weights=None, include_top=False,
                  input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    conv_base.trainable = True  # trainable VGG16 base

    model = Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(units=3, activation="softmax"))
    if summary :
        print(model.summary())
    return model