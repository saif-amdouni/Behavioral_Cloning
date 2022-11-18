import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from shared.CONFIG import *

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model import create_VGG16base

if len(tf.config.experimental.list_physical_devices('GPU')) > 0 :
    print("using GPU : ", tf.config.experimental.list_physical_devices('GPU'))
else : 
    # Set CPU as available physical device
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
    print("using CPU : ", tf.config.experimental.list_physical_devices('CPU'))

train = pd.read_csv(os.path.join(data_dir,csv))
train["target"] = train["target"].astype(str).map(target_map)

show_train_dist = False
if show_train_dist:
    plt.figure(figsize=(9, 8))
    sns.countplot(x="target", data=train, )
# create training and validation data
train_df, validate_df = train_test_split(train, test_size=0.1, stratify=train.target)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
print(f"total train : {total_train}")
print(f"total validate : {total_validate}")

# create training data generator
train_datagen = ImageDataGenerator(
    rotation_range=30,
    rescale=1. / 255,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False
)
train_generator = train_datagen.flow_from_dataframe(
    
    train_df,
    x_col='path',
    y_col='target',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    class_mode='categorical',
    batch_size=batch_size
)

# create Validation data generator

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    x_col='path',
    y_col='target',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    class_mode='categorical',
    batch_size=batch_size
)

# create model Modeling

model = create_VGG16base()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5), metrics=['accuracy'])



NAME = 'VGG16base-{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
log_dir = os.path.join("logs", NAME)
checkpoint_path = os.path.join(models_dir,"VGG16base_building", "cp-{epoch:04d}")

# create callbackss
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch',
    options=None
)

callbacks = [earlystop, learning_rate_reduction, tensorboard_callback, Checkpoint]

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size,
    steps_per_epoch=total_train // batch_size,
    callbacks=callbacks,

)
# tensorboard --logdir=foo:E:\ML\pythonProject\logs
