import tensorflow as tf

import os
import shutil
import numpy as np

import DataLoader
import model
from model import MyModel1,MyModel2
from config import Config

import datetime
from imgaug import augmenters as iaa

train_dir = "/backup/yuxin/aortaData/20191108_41/train"
#val_dir = "/backup/home/yuxin/aortaData/20191108_41对/val"
#test_dir = "/backup/home/yuxin/aortaData/20191108_41对/test"
csv_path = "/backup/yuxin/aortaData/20191108_41对/df.csv"
anno_path = "/backup/yuxin/aortaData/label"

dataset_train = DataLoader.Dataset_png()
#dataset_test = DataLoader.Dataset()
#dataset_val = DataLoader.Dataset()

dataset_train.load_dataset(train_dir)
#dataset_test.load_dataset(test_dir)
#dataset_val.load_dataset(val_dir)

augment = iaa.SomeOf((0, 2), [iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270)])])

case_ids_sort = sorted(dataset_train.case_id)
train_id = case_ids_sort[:66]
val_id = case_ids_sort[66:]

def train_generator():
    #assert isinstance(dataset, Dataset), "dadaset is not belong to Dataset class"
    for i in dataset_train.case_id:
        img, class_id = dataset_train.load_case_image(i, IMG_ID)
        det = augment.to_deterministic()
        img = det.augment_image(img)
        yield img, class_id
def train_gen():
        #assert isinstance(dataset, Dataset), "dadaset is not belong to Dataset class"
     for i in train_id:
        attr, case, case_anno, class_id = dataset_train.load_case(i, csv_path, anno_path, augmentation=augment)
        #inputs = [attr, case]
        yield {"input_1": case,"input_2": attr,"input_3":case_anno}, class_id

def val_gen():
    #assert isinstance(dataset, Dataset), "dadaset is not belong to Dataset class"
    for i in val_id:
        attr, case, case_anno, class_id = dataset_train.load_case(i, csv_path, anno_path)
        #inputs = [attr, case]
        yield {"input_1": case,"input_2": attr,"input_3":case_anno}, class_id

#def test_gen():
#    #assert isinstance(dataset, Dataset), "dadaset is not belong to Dataset class"
#     for i in dataset_test.case_id:
#         attr, case, class_id = dataset_test.load_case(i, csv_path)
#        yield {"input_1": case,"input_2": attr}, class_id

train_d = tf.data.Dataset.from_generator(
        train_gen, output_types=({"input_1": tf.float32,"input_2": tf.float32,"input_3": tf.float32}, tf.int8), 
        output_shapes=({"input_1": tf.TensorShape([14, Config.IMAGE_DIM, Config.IMAGE_DIM, 3]),"input_2": tf.TensorShape(18,),
             "input_3":tf.TensorShape([14,4,4])}, 
            tf.TensorShape([])))

val_d = tf.data.Dataset.from_generator(
        val_gen, output_types=({"input_1": tf.float32,"input_2": tf.float32,"input_3": tf.float32}, tf.int8), 
        output_shapes=({"input_1": tf.TensorShape([14, Config.IMAGE_DIM, Config.IMAGE_DIM, 3]),"input_2": tf.TensorShape(18,),
            "input_3":tf.TensorShape([14,4,4])},
            tf.TensorShape([])))

#test_d = tf.data.Dataset.from_generator(
#        test_gen, output_types=({"input_1": tf.float32,"input_2": tf.float32}, tf.int8), 
#        output_shapes=({"input_1": tf.TensorShape([14, Config.IMAGE_DIM, Config.IMAGE_DIM, 3]),"input_2": tf.TensorShape(18,)}, 
#            tf.TensorShape([])))

#train_d = train_d.shuffle()
train_d = train_d.batch(1)

#val_d = val_d.shuffle()
val_d = val_d.batch(1)

#test_d = test_d.shuffle(buffer_size=len(dataset_test.case_id))
#test_d = test_d.batch(1)

model = MyModel2(2, "vgg")
#model.keras_model.load_weights("/Extended/backup1/yuxin/weights/aorta/20200502T2332_vgg/aorta_0099.h5")
#model.train(train_d, 20, val_d, trainable=False)
model.test(val_d, "/Extended/backup1/yuxin/weights/aorta/20200502T2332_vgg/aorta_0099.h5")
#model.summary()
