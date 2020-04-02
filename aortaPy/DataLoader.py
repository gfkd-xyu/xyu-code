import tensorflow as tf
#from tensorflow import keras
#from tensorlfow.keras import layers

import SimpleITK as sitk
import pydicom
import cv2
import skimage.color as sc

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

from config import Config
from imgaug import augmenters as iaa
#import matplotlib.pyplot as plt
#print(tf.version)

def normalize(img):
        normalized_img = ((img - np.min(img))/(np.max(img) - np.min(img)))*255
        return normalized_img.astype(np.uint8)


class Dataset(object):
    
    
    def __init__(self, class_map=None):
        self.image_info = {}
        self.image_id = []
        self.case_info = {}
        self.case_id = []

    def add_case(self, case_id, path, class_id):
        if self.case_info.get(case_id) is not None: return
        #does the case_id exist already?
        #case infomation form [path, class_id]

        case_i = [path, class_id]
        case_info = {
            case_id : case_i
        }
        self.case_info.update(case_info)
        self.case_id.append(case_id)

    def add_image(self, image_id, path, case_id, class_id):
        if self.image_info.get(image_id) is not None: return
        image_i = [case_id, path, class_id]
        image_info = {
            image_id : image_i
        }
        self.image_info.update(image_info)
        self.image_id.append(image_id)

    def load_image(self, image_id):
        path = self.image_info[image_id][1]
        ds = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(ds)[0]
        #if img.ndim != 3:
        #    img = sc.gray2rgb(img)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img, self.image_info[image_id][2]

    def load_case_image(self, case_id, img_id):
        path = os.path.join(self.case_info[case_id][0],"{img_id}.dcm".format(img_id=str(img_id)))
        #ds = pydicom.read_file(path)
        #img = ds.pixel_array
        ds = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(ds)[0]
        img = normalize(img)
        #print(img.dtype)
        img = cv2.resize(img, (Config.IMAGE_DIM, Config.IMAGE_DIM), interpolation=cv2.INTER_AREA)
        if img.ndim != 3:
            img = sc.gray2rgb(img)
        #print(img.dtype)
        return img, self.case_info[case_id][1]

    def handle_case_attributes(self, csv_path):
        df = pd.read_csv(csv_path,header=0)
        df.drop(["tchol",'hdl','ldl','trig','hypotension'], axis=1, inplace=True)
        df['ane_diam'].fillna(0,inplace=True)
        df.fillna(method='ffill',inplace=True)
        small = LabelBinarizer().fit(df["smallbin"])
        df['smallbin']=small.transform(df['smallbin'])
        dia = LabelBinarizer().fit(df["diabetes"])
        hbp = LabelBinarizer().fit(df["HBP"])
        df['diabetes']=dia.transform(df['diabetes'])
        df['HBP']=hbp.transform(df['HBP'])
        cs = MinMaxScaler()
        df[df.columns[2:]] = cs.fit_transform(df[df.columns[2:]])
        df[df.columns[2:]] = df[df.columns[2:]].astype(np.float32)
        return df

    def load_case(self, case_id, csv_path, augmentation=None):
        case = []
        df = self.handle_case_attributes(csv_path)
        cid, group = case_id.split("_")
        if group=='event':
            attr = df[df.columns[2:]][(df['id']==int(cid))&(df['group']==1)]  
        else:
            attr = df[df.columns[2:]][(df['id']==int(cid))&(df['group']==0)]
        attr = np.array(attr).reshape(18)
        #case.append(np.array(attr))
        path = self.case_info[case_id][0]
        if augmentation is not None: det = augmentation.to_deterministic()
        for i in os.listdir(path):
            if i.endswith(".dcm"):
                ds = sitk.ReadImage(os.path.join(path,i))
                m = sitk.GetArrayFromImage(ds)[0]
                m = normalize(m)
                m = cv2.resize(m, (Config.IMAGE_DIM, Config.IMAGE_DIM), interpolation=cv2.INTER_AREA)
                if m.ndim != 3:
                    m = sc.gray2rgb(m)
                if augmentation is not None: m = det.augment_image(m)
                case.append(m)
        #case = np.stack(case,axis=-1)
        #case = np.expand_dims(case,axis=-1)
        return attr, np.array(case), self.case_info[case_id][1]
    
    #def prepare(self):
    def load_dataset(self, dataset_dir):
        print(dataset_dir)
        for d in os.listdir(dataset_dir):
            if d.startswith("."): continue
            if not os.path.isdir(os.path.join(dataset_dir, d)): continue
            #d is a dir or not
            tp = os.path.join(dataset_dir, d)
            for i in os.listdir(tp):
                if i.startswith("."): continue
                path = os.path.join(tp, i, "血管分叉层面")
                case_id = i
                class_id = 1 if "control" in i else 0  
                self.add_case(case_id, path, class_id)
                for j in os.listdir(path):
                    if j.endswith(".dcm"):
                        p = os.path.join(path,j)
                        self.add_image(i+"_"+os.path.splitext(j)[0], p, case_id, class_id)


#def data_generator(dataset, augment=False):
#    assert isinstance(dataset, Dataset), "dadaset is not belong to Dataset class"
#    for i in dataset.case_id:
#        case, class_id = dataset.load_case(i)
#        yield tf.convert_to_tensor(case), tf.convert_to_tensor(class_id)





