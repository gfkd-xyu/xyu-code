import tensorflow as tf

import os
import shutil
import numpy as np

import DataLoader
import model
from model import MyModel1
from config import Config

import datetime
from imgaug import augmenters as iaa

train_dir = "/aortaData/20191108_41对/train"
val_dir = "/aortaData/20191108_41对/val"
test_dir = "/backup/yuxin/aortaData/20191108_41对/test"
csv_path = "/backup/yuxin/aortaData/20191108_41对/df.csv"

dataset_train = DataLoader.Dataset()
dataset_test = DataLoader.Dataset()
dataset_val = DataLoader.Dataset()

dataset_train.load_dataset(train_dir)
dataset_test.load_dataset(test_dir)
dataset_val.load_dataset(val_dir)

augment = iaa.SomeOf((0, 2), [iaa.Fliplr(0.5),
    aa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270)])])

def train_generator():
    #assert isinstance(dataset, Dataset), "dadaset is not belong to Dataset class"
    for i in dataset_train.case_id:
        img, class_id = dataset_train.load_case_image(i, IMG_ID)
        det = augment.to_deterministic()
        img = det.augment_image(img)
        yield img, class_id

for i in range(14):
    IMG_ID = i+1
    train_dataset = tf.data.Dataset.from_generator(
            train_generator, (tf.float32, tf.int8), 
            (tf.TensorShape([Config.IMAGE_DIM, Config.IMAGE_DIM, 3]), tf.TensorShape([])))

    train_dataset = train_dataset.shuffle(buffer_size=len(dataset_train.case_id))
    train_dataset = train_dataset.batch(1)
    
    model = MyModel1(1)
    model.train(train_dataset, IMG_ID, 20, trainable=False)
    model.train(train_dataset, IMG_ID, 50, trainable=True)
    model.summary()
