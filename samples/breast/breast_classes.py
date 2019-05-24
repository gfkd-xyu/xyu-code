"""
Mask R-CNN
CBIS-DDSM

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import SimpleITK as sitk
import pandas as pd
from skimage import img_as_ubyte
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/breast/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
# VAL_IMAGE_IDS = 


############################################################
#  Configurations
############################################################

class BreastConfig(Config):
    """Configuration for training on the breast segmentat
    ion dataset."""
    # Give the configuration a recognizable name
    NAME = "breast"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (874 - 262) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, 262 // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_CHANNEL_COUNT = 3
    IMAGE_MIN_SCALE = 0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000 
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    
    # Grayscale images
    # IMAGE_CHANNEL_COUNT = 1
    # Image mean (Grayscale)
    MEAN_PIXEL = np.array([32768.0, 32768.0, 32768.0])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 10
    LEARNING_RATE = 0.0001

class BreastInferenceConfig(BreastConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    # IMAGE_RESIZE_MODE = "square"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.5
    DETECTION_MIN_CONFIDENCE = 0.5

############################################################
#  Dataset
############################################################

class BreastDataset(utils.Dataset):

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
        if info["source"] == "breast":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = BreastDataset()
    dataset_train.load_breast(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BreastDataset()
    dataset_val.load_breast(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                augmentation=augmentation,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = BreastDataset()
    dataset.load_breast(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
    	# molded_images
        #h,w = image.shape[:2]
        #scale = 512 / max(h,w)
        #image = utils.resize(image,(round(h*scale),round(w*scale)),preserve_range=True)      
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # image = skimage.color.rgb2gray(image)
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        print(dataset.image_info[image_id]["id"])
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

############################################################
#  Evaluation
############################################################

def evaluate(model, dataset_dir, subset, config):
    """Run evaluation on images in the given directory."""
    print("Running on images in the given directory")

    dataset = BreastDataset()
    dataset.load_breast(dataset_dir, subset)
    dataset.prepare()

    image_ids = dataset.image_ids
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset, config,
                        image_id, use_mini_mask=False)
        # molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        print(image_id,dataset.image_info[image_id]["id"])
        #print(gt_mask.shape,r['masks'].shape)       
        try:
            AP, precisions, recalls, overlaps =\
                    utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'])
        except:
            print(image_id,dataset.image_info[image_id]["id"])
            print(gt_mask.shape,r['masks'].shape)
            #continue
            break
        APs.append(AP)
        #recallss.append(recalls)
        #   overlapss.append(overlaps)
    print("mAP: ", np.mean(APs))
    #print("recall: ", recallss)
    #print("overlap: ", np.mean(overlapss))

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect' or 'evaluate'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect" or args.command == "evaluate":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    if args.command == "train":
        CSV_DIR = "/backup/yuxin/mass_case_description_train_set.csv"
    else:
        CSV_DIR = "/backup/yuxin/mass_case_description_test_set.csv"
    # Configurations
    if args.command == "train":
       
        config = BreastConfig()
    else:
        config = BreastInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "evaluate":
        evaluate(model, args.dataset, args.subset, config)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
