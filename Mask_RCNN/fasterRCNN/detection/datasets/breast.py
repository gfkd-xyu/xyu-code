import os.path as osp
import os
import pandas as pd
import cv2
import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import skimage.io
import logging
import random

from detection.datasets import transforms, utils

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.image_case = {}
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def add_case(self, case_id, image_id):
        if self.image_case.get(case_id) is not None:
            self.image_case[case_id].append(image_id)
        image_case = {
                case_id: [image_id]
                }
        self.image_case.update(image_case)

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        #prepare image_case 
        for info, id in zip(self.image_info, self.image_ids):
            case_id = "_".join(info["id"].split("_")[2:4])
            self.add_case(case_id, id)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]
    
    def normalize(self, img):
        normalized_img = ((img - np.min(img))/(np.max(img) - np.min(img)))*255
        return normalized_img


    def load_breast(self, dataset_dir, subset):
        """Load a subset of the breast dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        # self.add_class("breast", 1, "NORMAL")
        self.add_class("breast", 1, "BENIGN")
        self.add_class("breast", 2, "MALIGNANT")
        df = pd.read_csv(CSV_DIR)
        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test"]
        subset_dir = "train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "test":
            image_ids = []
            for d in os.listdir(dataset_dir):
                _, p, p_id, rl, iv=d.split("_")
                for f in os.listdir(os.path.join(dataset_dir,d,"masks")):
                    # x = df[(df['patient_id']=="P_"+p_id)&(df['left or right breast']==rl)&
                    #        (df['image view']==iv)&(df['abnormality id']==int(f[0]))]['pathology'].values[0]
                    bd = df[(df['patient_id']=="P_"+p_id)&(df['left or right breast']==rl)&
                            (df['image view']==iv)&(df['abnormality id']==int(f[0]))]['breast_density'].values[0]
                    if bd==2 or bd==1:
                        image_ids.append(d)
            #image_ids = os.listdir(dataset_dir)
        else:
            x = []
            for d in os.listdir(dataset_dir):
                _, p, p_id, rl, iv=d.split("_")
                for f in os.listdir(os.path.join(dataset_dir,d,"masks")):
                    # x = df[(df['patient_id']=="P_"+p_id)&(df['left or right breast']==rl)&
                    #        (df['image view']==iv)&(df['abnormality id']==int(f[0]))]['pathology'].values[0]
                    bd = df[(df['patient_id']=="P_"+p_id)&(df['left or right breast']==rl)&
                            (df['image view']==iv)&(df['abnormality id']==int(f[0]))]['breast_density'].values[0]
                    if bd==2 or bd==1:
                        x.append(d)
            #x = os.listdir(dataset_dir)
            y = np.ones(len(x))
            train_x, val_x, _, _, = train_test_split(x, y, test_size=0.3, random_state=7)
            if subset == "val":
                image_ids = val_x
            else:
                # Get image ids from directory names
                image_ids = train_x
        

        # Add images
        for image_id in image_ids:
            self.add_image(
                "breast",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/000000.png"))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        # get image class_id
        _, _, pt_id, lr, iv = info["id"].split("_")
        df = pd.read_csv(CSV_DIR)
        # Read mask files from .png image
        mask = []
        class_ids = []
        for f in os.listdir(mask_dir):
            if f.endswith(".png"):
                #ds = sitk.ReadImage(os.path.join(mask_dir, f))
                #m = sitk.GetArrayFromImage(ds)
                #m = np.squeeze(m)
                #m = m.astype(np.bool)
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
                #print(pt_id,f,df[(df['patient_id']=="P_"+pt_id)&(df['left or right breast']==lr)&(df['image view']==iv)&(df['abnormality id']==int(f[0]))]['pathology'].values)
                class_name = df[(df['patient_id']=="P_"+pt_id)&(df['left or right breast']==lr)&(df['image view']==iv)&(df['abnormality id']==int(f[0]))]['pathology'].values[0]
                class_id = 2 if class_name=='MALIGNANT' else 1
                class_ids.append(class_id)
        mask = np.stack(mask, axis=-1)
        class_ids = np.array(class_ids)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, class_ids.astype(np.int32)  #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["id"]

    
    def load_image_gt(self, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
        """Load and return ground truth data for an image (image, mask, bounding boxes).

        augment: (deprecated. Use augmentation instead). If true, apply random
            image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.
        use_mini_mask: If False, returns full-size masks that are the same height
            and width as the original image. These can be big, for example
            1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
            224x224 and are generated by extracting the bounding box of the
            object and resizing it to MINI_MASK_SHAPE.

        Returns:
        image: [height, width, 3]
        shape: the original shape of the image before resizing and cropping.
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. The height and width are those
            of the image unless use_mini_mask is True, in which case they are
            defined in MINI_MASK_SHAPE.
        """
        # Load image and mask
        image = self.load_image(image_id)
        mask, class_ids = self.load_mask(image_id)
        original_shape = image.shape
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        # print(image.shape,mask.shape)
        mask = utils.resize_mask(mask, scale, padding, crop)
        # Random horizontal flips.
        # TODO: will be removed in a future update in favor of augmentation
        if augment:
            logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
            if random.randint(0, 1):
                image = np.fliplr(image)
                mask = np.fliplr(mask)

        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if augmentation:
            import imgaug

            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                            "Fliplr", "Flipud", "CropAndPad",
                            "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image = det.augment_image(image)
            #print(image.shape)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask = det.augment_image(mask.astype(np.uint8))
            #                        hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            #mask = mask.astype(np.bool)

        # Note that some boxes might be all zeros if the corresponding mask got cropped out.
        # and here is to filter them out
        _idx = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, _idx]
        class_ids = class_ids[_idx]
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox = utils.extract_bboxes(mask)

        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        active_class_ids = np.zeros([self.num_classes], dtype=np.int32)
        source_class_ids = self.source_class_ids[self.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1

        # Resize masks to smaller size to reduce memory usage
        if use_mini_mask:
            mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

        # Image meta data
        img_meta_dict = dict({
            'ori_shape': original_shape,
            'img_shape': ,
            'pad_shape': image.shape,
            'scale_factor': scale,
            'flip': flip
        })
        img_meta = utils.compose_image_meta(img_meta_dict)
        image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                        window, scale, active_class_ids)

        return image, image_meta, class_ids, bbox, mask   

    # def compose_image_meta(image_id, original_image_shape, image_shape,
    #                         window, scale, active_class_ids):
    #     """Takes attributes of an image and puts them in one 1D array.

    #     image_id: An int ID of the image. Useful for debugging.
    #     original_image_shape: [H, W, C] before resizing or padding.
    #     image_shape: [H, W, C] after resizing and padding
    #     window: (y1, x1, y2, x2) in pixels. The area of the image where the real
    #             image is (excluding the padding)
    #     scale: The scaling factor applied to the original image (float32)
    #     active_class_ids: List of class_ids available in the dataset from which
    #         the image came. Useful if training on images from multiple datasets
    #         where not all classes are present in all datasets.
    #     """
    #     meta = np.array(
    #         [image_id] +                  # size=1
    #         list(original_image_shape) +  # size=3
    #         list(image_shape) +           # size=3
    #         list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
    #         [scale] +                     # size=1
    #         list(active_class_ids)        # size=num_classes
    #     )
    #     return meta


class CocoDataSet(object):
    def __init__(self, dataset_dir, subset,
                 flip_ratio=0,
                 pad_mode='fixed',
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 scale=(1024, 800),
                 debug=False):
        '''Load a subset of the COCO dataset.
        
        Attributes
        ---
            dataset_dir: The root directory of the COCO dataset.
            subset: What to load (train, val).
            flip_ratio: Float. The ratio of flipping an image and its bounding boxes.
            pad_mode: Which padded method to use (fixed, non-fixed)
            mean: Tuple. Image mean.
            std: Tuple. Image standard deviation.
            scale: Tuple of two integers.
        '''
        
        if subset not in ['train', 'val']:
            raise AssertionError('subset must be "train" or "val".')
            

        self.coco = COCO("{}/annotations/instances_{}2017.json".format(dataset_dir, subset))

        # get the mapping from original category ids to labels
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        
        self.img_ids, self.img_infos = self._filter_imgs()
        
        if debug:
            self.img_ids, self.img_infos = self.img_ids[:50], self.img_infos[:50]
            
        self.image_dir = "{}/{}2017".format(dataset_dir, subset)
        
        self.flip_ratio = flip_ratio
        
        if pad_mode in ['fixed', 'non-fixed']:
            self.pad_mode = pad_mode
        elif subset == 'train':
            self.pad_mode = 'fixed'
        else:
            self.pad_mode = 'non-fixed'
        
        self.img_transform = transforms.ImageTransform(scale, mean, std, pad_mode)
        self.bbox_transform = transforms.BboxTransform()
        
        
    def _filter_imgs(self, min_size=32):
        '''Filter images too small or without ground truths.
        
        Args
        ---
            min_size: the minimal size of the image.
        '''
        # Filter images without ground truths.
        all_img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))
        # Filter images too small.
        img_ids = []
        img_infos = []
        for i in all_img_ids:
            info = self.coco.loadImgs(i)[0]
            
            ann_ids = self.coco.getAnnIds(imgIds=i)
            ann_info = self.coco.loadAnns(ann_ids)
            ann = self._parse_ann_info(ann_info)
            
            if min(info['width'], info['height']) >= min_size and ann['labels'].shape[0] != 0:
                img_ids.append(i)
                img_infos.append(info)
        return img_ids, img_infos
        
    def _load_ann_info(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info

    def _parse_ann_info(self, ann_info):
        '''Parse bbox annotation.
        
        Args
        ---
            ann_info (list[dict]): Annotation info of an image.
            
        Returns
        ---
            dict: A dict containing the following keys: bboxes, 
                bboxes_ignore, labels.
        '''
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)


        return ann
    
    def __len__(self):
        return len(self.img_infos)
    
    def __getitem__(self, idx):
        '''Load the image and its bboxes for the given index.
        
        Args
        ---
            idx: the index of images.
            
        Returns
        ---
            tuple: A tuple containing the following items: image, 
                bboxes, labels.
        '''
        img_info = self.img_infos[idx]
        ann_info = self._load_ann_info(idx)
        
        # load the image.
        img = cv2.imread(osp.join(self.image_dir, img_info['file_name']), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        ori_shape = img.shape
        
        # Load the annotation.
        ann = self._parse_ann_info(ann_info)
        bboxes = ann['bboxes']
        labels = ann['labels']
        
        flip = True if np.random.rand() < self.flip_ratio else False
        
        # Handle the image
        img, img_shape, scale_factor = self.img_transform(img, flip)

        pad_shape = img.shape
        
        # Handle the annotation.
        bboxes, labels = self.bbox_transform(
            bboxes, labels, img_shape, scale_factor, flip)
        
        # Handle the meta info.
        img_meta_dict = dict({
            'ori_shape': ori_shape,
            'img_shape': img_shape,
            'pad_shape': pad_shape,
            'scale_factor': scale_factor,
            'flip': flip
        })

        img_meta = utils.compose_image_meta(img_meta_dict)
        
        return img, img_meta, bboxes, labels
    
    def get_categories(self):
        '''Get list of category names. 
        
        Returns
        ---
            list: A list of category names.
            
        Note that the first item 'bg' means background.
        '''
        return ['bg'] + [self.coco.loadCats(i)[0]["name"] for i in self.cat2label.keys()]

