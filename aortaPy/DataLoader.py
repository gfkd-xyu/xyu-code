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
from xml.etree import ElementTree as ET

from config import Config
from imgaug import augmenters as iaa
import imgaug as ia
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

   
    def load_case(self, case_id, csv_path, anno_path, augmentation=None):
        case = []
        case_anno = []
        df = self.handle_case_attributes(csv_path)
        cid, group = case_id.split("_")
        class_id = 1 if group=='event' else 0
        attr = df[df.columns[2:]][(df['id']==int(cid))&(df['group']==class_id)]  
        attr = np.array(attr).reshape(18)
        #case.append(np.array(attr))
        path = self.case_info[case_id][0]
        if augmentation is not None: det = augmentation.to_deterministic()
        #for i in os.listdir(path):
        for i in range(1,15):
            xml_file = os.path.join(anno_path, str(i), str(class_id), "{}.xml".format(cid))
            #print(xml_file)
            file = "{}.dcm".format(i)
            ds = sitk.ReadImage(os.path.join(path,file))
            #print(os.path.join(path,file))
            m = sitk.GetArrayFromImage(ds)[0]
            m = normalize(m)
            m = cv2.resize(m, (Config.IMAGE_DIM, Config.IMAGE_DIM), interpolation=cv2.INTER_AREA)
            if m.ndim != 3:
                m = sc.gray2rgb(m)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            all_boxes = []
            for o in root.iter('object'):
                for b in o.findall('bndbox'):
                    ymin = int(b.find("ymin").text)
                    xmin = int(b.find("xmin").text)
                    ymax = int(b.find("ymax").text)
                    xmax = int(b.find("xmax").text)
                    #bbs = [xmin, ymin, xmax, ymax]
                    bbs =ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
                    all_boxes.append(bbs)
            bbs_aug = ia.BoundingBoxesOnImage(all_boxes,shape=m.shape)
            if augmentation is not None: 
                m = det.augment_image(m)
                bbs_aug = det.augment_bounding_boxes(bbs_aug)
      
            case.append(m)
            case_anno.append(bbs_aug)
        #case = np.stack(case,axis=-1)
        #case = np.expand_dims(case,axis=-1)
        return attr, np.array(case), np.array(case_anno), self.case_info[case_id][1]
    
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

class Dataset_png(object):
    
    def __init__(self, class_map=None):

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

   
    def load_case(self, case_id, csv_path, anno_path, augmentation=None):
        case = []
        case_anno = []
        df = self.handle_case_attributes(csv_path)
        cid, class_id = case_id.split("_")
        attr = df[df.columns[2:]][(df['id']==int(cid))&(df['group']==int(class_id))]  
        attr = np.array(attr).reshape(18)
        #case.append(np.array(attr))
        path = self.case_info[case_id][0]
        if augmentation is not None: det = augmentation.to_deterministic()
        #for i in os.listdir(path):
        #cont = 0
        for i in range(1,15):
            xml_file = os.path.join(anno_path, str(i), str(class_id), cid+".xml")
            #print(xml_file)
            img_file = os.path.join(path, str(i), str(class_id), cid+".png")
            m = cv2.imread(img_file, cv2.IMREAD_ANYDEPTH)
            m = cv2.resize(m, (Config.IMAGE_DIM, Config.IMAGE_DIM), interpolation=cv2.INTER_AREA)
            if m.ndim != 3:
                m = sc.gray2rgb(m)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            all_boxes = []
            for o in root.iter('object'):
                for b in o.findall('bndbox'):
                    ymin = int(b.find("ymin").text)
                    xmin = int(b.find("xmin").text)
                    ymax = int(b.find("ymax").text)
                    xmax = int(b.find("xmax").text)
                    #bbs = [xmin, ymin, xmax, ymax]
                    bbs =ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
                    all_boxes.append(bbs)
            bbs_aug = ia.BoundingBoxesOnImage(all_boxes,shape=m.shape)
            if augmentation is not None: 
                m = det.augment_image(m)
                bbs_aug = det.augment_bounding_boxes(bbs_aug)
            bboxes = np.zeros([4, 4],dtype=np.int32)
            #cont = max(cont, len(all_boxes))
            for j in range(len(bbs_aug.bounding_boxes)):
                n_x1 = bbs_aug.bounding_boxes[j].x1
                n_y1 = bbs_aug.bounding_boxes[j].y1
                n_x2 = bbs_aug.bounding_boxes[j].x2
                n_y2 = bbs_aug.bounding_boxes[j].y2
                bboxes[j] = np.array([n_y1, n_x1, n_y2, n_x2])
            case.append(m)
            case_anno.append(bboxes.astype(np.int32))
        #case = np.stack(case,axis=-1)
        #case = np.expand_dims(case,axis=-1)
        #print(cont)
        return attr, np.array(case), np.array(case_anno), self.case_info[case_id][1]
    
    #def prepare(self):
    def load_dataset(self, dataset_dir):
        print(dataset_dir)
            #d is a dir or not
        tp = os.path.join(dataset_dir, '1', "1")
        for j in os.listdir(tp):
            if j.endswith(".png"):
                #path = os.path.join(tp,j)
                class_id = 1
                case_id = os.path.splitext(j)[0]+"_1"
                self.add_case(case_id, dataset_dir, class_id)
        tp = os.path.join(dataset_dir, '1', "0")
        for j in os.listdir(tp):
            if j.endswith(".png"):
                #path = os.path.join(tp,j)
                class_id = 0
                case_id = os.path.splitext(j)[0]+"_0"
                self.add_case(case_id, dataset_dir, class_id)       



#def data_generator(dataset, augment=False):
#    assert isinstance(dataset, Dataset), "dadaset is not belong to Dataset class"
#    for i in dataset.case_id:
#        case, class_id = dataset.load_case(i)
#        yield tf.convert_to_tensor(case), tf.convert_to_tensor(class_id)





