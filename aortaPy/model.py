import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, Input
#from tfkerassurgeon import delete_layer, insert_layer

import numpy as np
import os 
import shutil
import datetime
print(tf.__version__)

from config import Config
from non_local import non_local_block

import pydot 
pydot.find_graphviz = lambda: True 
from tensorflow.keras.utils import plot_model 
from detection.models.necks import fpn
from detection.models.roi_extractors import roi_align

"""
# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
"""
#from __future__ import print_function
#from __future__ import absolute_import
#from __future__ import division

#import os
import warnings

#from . import get_submodules_from_kwargs
#from . import imagenet_utils
#from .imagenet_utils import decode_predictions
#from .imagenet_utils import _obtain_input_shape


BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')

#backend = None
#layers = None
#models = None
#keras_utils = None


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000,
              **kwargs):

#    global backend, layers, models, keras_utils
#    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

#    input_shape = _obtain_input_shape(input_shape,
#                                      default_size=default_size,
#                                      min_size=32,
#                                      data_format=backend.image_data_format(),
#                                      require_flatten=include_top,
#                                      weights=weights)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            warnings.warn('`input_shape` is undefined or non-square, '
                          'or `rows` is not in [128, 160, 192, 224]. '
                          'Weights for input shape (224, 224) will be'
                          ' loaded as the default.')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    C1 = x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    C2 = x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    C3 = x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    C4 = x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    C5 = x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if backend.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(dropout, name='dropout')(x)
        x = layers.Conv2D(classes, (1, 1),
                          padding='same',
                          name='conv_preds')(x)
        x = layers.Reshape((classes,), name='reshape_2')(x)
        x = layers.Activation('softmax', name='act_softmax')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tf.keras.models.Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    # Load weights.
    if weights == 'imagenet':
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = tf.keras.utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = tf.keras.utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

class BatchNorm(tf.keras.layers.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)



def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = tf.keras.layers.Add()([x, input_tensor])
    x = tf.keras.layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = tf.keras.layers.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = tf.keras.layers.ZeroPadding2D((3, 3))
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    C1 = x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
        return C5
    else:
        C5 = None
        return C4
    return [C1, C2, C3, C4, C5]

def Aorta_Model(img_shape=None, attr_shape=None, pre_trained='resnet', classes=2, trainable=False, **kwargs):
    img_input = layers.Input(shape=img_shape)

    attr_input = layers.Input(shape=attr_shape)
 
    #channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    if pre_trained=="resnet":
        pre_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    else:
        pre_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

    pre_model.trainable = trainable

    
    #attr_x = layers.Reshape(18,)(attr_input)
    feature = []
    feature.append(attr_input)
    for i in range(14):
        print(img_input[:,i,:,:,:].shape)
        x = pre_model(img_input[:,i,:,:,:])
        x = layers.Flatten(x)
        x = layers.Dense(1000, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        feature.append(x)
    x = layers.concatenate(feature)
    print(x.shape)
    x = layers.Dense(classes, activation='softmax')(x)

    inputs = [img_input, attr_input]
    model = tf.keras.Model(inputs, x, name='aorta_model')

    return model


class MyModel2(tf.keras.Model):
    def __init__(self, classes, pre_trained):
        super().__init__()
        #self.conv0 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')
        #self.act0 = tf.keras.layers.Activation('relu')
        #self.mpl0 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        #self.conv1 = tf.keras.layers.Conv2D(32, (3, 3))
        #self.act1 = tf.keras.layers.Activation('relu')
        #self.mpl1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        #self.resnet50 = self.load_resnet()
        #self.vgg = self.load_vgg()
        #self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.pool_size = (7, 7)
        self.neck = [fpn.FPN(name='fpn')]*14
        self.roi_align = [roi_align.PyramidROIAlign(
            pool_shape=self.pool_size,
            name='pyramid_roi_align')]*14
        #self.backbone = resnet.ResNet(depth=101, name='res_net')
        self.classes = classes
        self.epoch = 0
        self.pre_trained = pre_trained 
        self.dense0 =layers.Dense(4096, activation='relu', name='fc1')
        self.dense1 =layers.Dense(100, activation='relu', name='fc2')
        #self.xception = tf.keras.applications.Xception(include_top=False, weights='imagenet') 
        #self.inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        if self.pre_trained == "resnet":
            self.pre_model = [tf.keras.applications.ResNet50(include_top=False, weights='imagenet')]*14
        elif self.pre_trained == "vgg":
            self.pre_model = [tf.keras.applications.VGG19(include_top=False, weights='imagenet')]*14
        elif self.pre_trained == "xception":
            self.pre_model = [tf.keras.applications.Xception(include_top=False, weights='imagenet')]*14
        elif self.pre_trained == "inception":
            self.pre_model = [tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')]*14
        elif self.pre_trained == "mobile":
            self.pre_model = [tf.keras.applications.MobileNet(include_top=False,weights='imagenet')]*14

        self.dense = tf.keras.layers.Dense(classes, activation='sigmoid')
        self.keras_model = self.build()
        #self.resnet = resnet_graph("resnet50",stage5=True)J
    
    def build(self):
        #x = self.conv0(inputs)
        #x = self.act0(x)
        #x = self.mpl0(x)
        
        #x = self.conv1(x)
        #x = self.act1(x)
        #x = self.mpl1(x)
        #x = self.load_resnet(x)
        img_input = layers.Input(shape=(14, Config.IMAGE_DIM, Config.IMAGE_DIM, 3),name="input_1")
        attr_input = layers.Input(shape=(18,),name="input_2")
        anno_input = layers.Input(shape=(14,4,4),name="input_3")
        #x = self.vgg(inputs)
        #x = self.gap(x)
        #x = self.flt(x)
        #x = self.dense0(x)
        #x = layers.Dropout(0.5)(x)
        #x = self.dense1(x)
        #x = layers.Dropout(0.5)(x)
        #x = self.dense(x)
        #model = tf.keras.Model(inputs,x)
        #return model
        #attr_x = layers.Reshape(18,)(attr_input)
        feature = []
        feature.append(attr_input)
        for i in range(14):
            print(img_input[:,i,:,:,:].shape)
            x = self.pre_model[i].layers[1](img_input[:,i,:,:,:])
            #print(x.shape)
            for l in self.pre_model[i].layers[2:]:
                if l.name == "conv_pw_3_relu":
                    C2 = x = l(x)
                elif l.name == "conv_pw_5_relu": 
                    C3 = x = l(x)
                elif l.name == "conv_pw_7_relu": 
                    C4 = x = l(x)
                elif l.name == "conv_pw_13_relu": 
                    C5 = x = l(x)
                else:
                    x = l(x)
                #print(x.shape)
            print(C2.shape, C3.shape, C4.shape, C5.shape)
            #if not tf.math.logical_and(C2,C3,C4,C5): raise ValueError('fpn layers has no value')
            #C2 = self.pre_model[i].get_layer('conv_pw_3_relu').output
            #C3 = self.pre_model[i].get_layer('conv_pw_5_relu').output
            #C4 = self.pre_model[i].get_layer('conv_pw_7_relu').output
            #C5 = self.pre_model[i].get_layer('conv_pw_13_relu').output
            P2, P3, P4, P5, _ = self.neck[i]([C2, C3, C4, C5], 
                                       training=True)
            feature_maps = [P2, P3, P4, P5]
            #print(P2.shape)
            #pooled_rois_list: list of [num_rois, pooled_height, pooled_width, channels].
            #    The width and height are those specific in the pool_shape in the layer
            #    constructor.
            print(anno_input[:,i,:,:].shape)
            y = self.roi_align[i]((anno_input[:,i,:,:], feature_maps), training=True)
            y = layers.TimeDistributed(layers.Conv2D(1024, self.pool_size, padding="valid", activation='relu'),
                    name="{}_class_conv1".format(i))(y)
            #print("x", x.shape)

            y = layers.TimeDistributed(layers.Conv2D(1024, (1, 1), activation='relu'),
                    name="{}_class_conv2".format(i))(y)

            y = layers.Lambda(lambda t: backend.squeeze(backend.squeeze(t, 3), 2),
                    name="{}_pool_squeeze".format(i))(y)

            y = layers.TimeDistributed(layers.Dense(1024, activation='relu'),name="{}_fc1".format(i))(y)
            print(y.shape)
            s = backend.int_shape(y)
            y = layers.Reshape((s[1]*s[2],),name="{}_reshape".format(i))(y)
            #y = layers.Lambda(lambda t:tf.reshape(t, [tf.shape(t)[0],-1]),name="{}_reshape".format(i))(y)
            y = layers.Dense(1024, activation='relu', name="{}_fc2".format(i))(y)
            x = layers.GlobalAveragePooling2D(name="{}_gap".format(i))(x)
            #x = layers.Dense(2048, activation='relu', name="{}_fc3".format(i))(x)
            #x = layers.Dropout(0.5, name="{}_dp1".format(i))(x)
            x = layers.Dense(1024, activation='relu', name="{}_fc4".format(i))(x)
            x = layers.Dropout(0.5, name="{}_dp2".format(i))(x)
            
            x = layers.concatenate([x,y], name="{}_ccn".format(i))
            x = layers.Dense(100, activation='sigmoid', name="{}_fc5".format(i))(x)
            print(x.shape)
            feature.append(x)
        
        x = layers.concatenate(feature)
        print(x.shape)
        #x = layers.Dense(23, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(self.classes, activation='softmax')(x)

        inputss = [img_input, attr_input, anno_input]
        model = tf.keras.Model(inputs=inputss, outputs=x, name='aorta_model')

        return model
   

    def train(self, train_dataset, epochs, val_dataset, trainable=False):
        now = datetime.datetime.now()
        model_dir = os.path.join('/backup/home/yuxin/Mask_RCNN/aortaPy/weights','aorta')
        log_dir = os.path.join(model_dir, "{:%Y%m%dT%H%M}_{}".format(now, self.pre_trained))
        checkpoint_path = os.path.join(log_dir, "aorta_*epoch*.h5")
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

        callbacks_list = [
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False),
                tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=True)]

        log("\nStarting at epoch {}. LR={}\n".format(self.epoch,0.001))
        log("Checkpoint Path: {}".format(checkpoint_path))
        self.pre_model.trainable = trainable

        self.keras_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])

        self.keras_model.fit(
                train_dataset,
                initial_epoch=self.epoch,
                epochs=epochs,
                callbacks=callbacks_list,
                validation_data=val_dataset,
                workers=8,
                use_multiprocessing=True
                )
        self.epoch = max(self.epoch, epochs)





class MyModel1(tf.keras.Model):
    def __init__(self, classes, pre_trained):
        super().__init__()
        #self.conv0 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')
        #self.act0 = tf.keras.layers.Activation('relu')
        #self.mpl0 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        #self.conv1 = tf.keras.layers.Conv2D(32, (3, 3))
        #self.act1 = tf.keras.layers.Activation('relu')
        #self.mpl1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        #self.resnet50 = self.load_resnet()
        #self.vgg = self.load_vgg()
        #self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.classes = classes
        self.epoch = 0
        self.pre_trained = pre_trained
        self.flt = tf.keras.layers.Flatten()
        self.dense0 =tf.keras.layers.Dense(4096, activation='relu', name='fc1')
        self.dense1 =tf.keras.layers.Dense(100, activation='relu', name='fc2')
        #self.xception = tf.keras.applications.Xception(include_top=False, weights='imagenet') 
        #self.inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        if self.pre_trained == "resnet":
            self.pre_model = [tf.keras.applications.ResNet50(include_top=False, weights='imagenet')]*14
        elif self.pre_trained == "vgg":
            self.pre_model = [tf.keras.applications.VGG19(include_top=False, weights='imagenet')]*14
        elif self.pre_trained == "xception":
            self.pre_model = [tf.keras.applications.Xception(include_top=False, weights='imagenet')]*14
        elif self.pre_trained == "inception":
            self.pre_model = [tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')]*14
        elif self.pre_trained == "mobile":
            self.pre_model = [tf.keras.applications.MobileNet(include_top=False,weights='imagenet')]*14
        self.dense = tf.keras.layers.Dense(classes, activation='sigmoid')
        self.keras_model = self.build()
        #self.resnet = resnet_graph("resnet50",stage5=True)
    
    def build(self):
        #x = self.conv0(inputs)
        #x = self.act0(x)
        #x = self.mpl0(x)
        
        #x = self.conv1(x)
        #x = self.act1(x)
        #x = self.mpl1(x)
        #x = self.load_resnet(x)
        input_1 = layers.Input(name='input_1',shape=(14, Config.IMAGE_DIM, Config.IMAGE_DIM, 3))
        input_2 = layers.Input(name='input_2',shape=(18,))
        #x = self.vgg(inputs)
        #x = self.gap(x)
        #x = self.flt(x)
        #x = self.dense0(x)
        #x = layers.Dropout(0.5)(x)
        #x = self.dense1(x)
        #x = layers.Dropout(0.5)(x)
        #x = self.dense(x)
        #model = tf.keras.Model(inputs,x)
        #return model
        #attr_x = layers.Reshape(18,)(attr_input)
        feature = []
        feature.append(input_2)
        for i in range(14):
            print(input_1[:,i,:,:,:].shape)
            x = self.pre_model[i](input_1[:,i,:,:,:])
            x = layers.GlobalAveragePooling2D(name="{}_gap".format(i))(x)
            x = layers.Dense(512, activation='relu',name="{}_fc1".format(i))(x)
            x = layers.Dropout(0.5)(x)
            #x = layers.Dense(100, activation='relu',name="{}_fc2".format(i))(x)
            #x = layers.Dropout(0.5)(x)
            x = layers.Dense(2, activation='sigmoid',name="{}_fc3".format(i))(x)
            feature.append(x)
        x = layers.concatenate(feature)
        print(x.shape)
        x = layers.Dense(40, activation='relu')(x)
        #x = layers.Dense(10, activation='relu')(x)
        x = layers.Dense(self.classes, activation='softmax')(x)

        inputss = [input_1, input_2]
        model = tf.keras.Model(inputs=inputss, outputs=x, name='aorta_model')

        return model
   
    def load_vgg(self):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', 
                input_tensor=tf.keras.Input(shape=(Config.IMAGE_DIM, Config.IMAGE_DIM, 3)))
        #vgg.trainable = True
        return vgg


    def load_resnet(self):
        resnet50 = tf.keras.applications.ResNet50(
            include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=(Config.IMAGE_DIM, Config.IMAGE_DIM, 3)))
        #resnet50.trainable = True
        return resnet50

    def train(self, train_dataset, epochs, val_dataset, trainable=False):
        now = datetime.datetime.now()
        model_dir = os.path.join('/backup/home/yuxin/Mask_RCNN/aortaPy/weights','aorta')
        log_dir = os.path.join(model_dir, "{:%Y%m%dT%H%M}_{}".format(now, self.pre_trained))
        checkpoint_path = os.path.join(log_dir, "aorta_*epoch*.h5")
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

        callbacks_list = [
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False),
                tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=True)]

        log("\nStarting at epoch {}. LR={}\n".format(self.epoch,0.001))
        log("Checkpoint Path: {}".format(checkpoint_path))
        self.pre_model.trainable = trainable

        self.keras_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])

        self.keras_model.fit(
                train_dataset,
                initial_epoch=self.epoch,
                epochs=epochs,
                callbacks=callbacks_list,
                validation_data=val_dataset,
                workers=8,
                use_multiprocessing=True
                )
        self.epoch = max(self.epoch, epochs)

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """""
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)
    

def DepthwiseNet(input_tensor=None, input_shape=None, classes=2, **kwargs):


    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3,3), strides=(2,2), use_bias=False, 
                                                    name='block1_dwconv1')(img_input)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_bn1')(x)
    x = layers.Activation('relu', name='block1_act1')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3,3), strides=(2,2), use_bias=False,
                                                    name='block1_dwconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_bn2')(x)
    x = layers.Activation('relu', name='block1_act2')(x)

    residual = layers.Conv2D(112, (1,1), strides=(2,2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3,3), padding='same',
                                                    use_bias=False, name='block2_dwconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_bn1')(x)
    x = layers.Activation('relu', name='block2_act1')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', 
                                                    use_bias=False, name='block2_dwconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_bn2')(x)
    x = layers.MaxPool2D((3,3), strides=(2,2), padding='same', name='block2_pool')(x)

    x = layers.add([x, residual])

    residual = layers.Conv2D(224, (1,1), strides=(2,2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_act1')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3,3), padding='same', 
                                                    use_bias=False, name='block3_dwconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_bn1')(x)
    x = layers.Activation('relu', name='block3_act2')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same',
                                                    use_bias=False, name='block3_dwconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_bn2')(x)
    x = layers.MaxPool2D((3,3), strides=(2,2), padding='same', name='block3_pool')(x)

    x = layers.add([x, residual])

    residual = layers.Conv2D(448, (1,1), strides=(2,2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_act1')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3,3), padding='same',
                                                    use_bias=False, name='block4_dwconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_bn1')(x)
    x = layers.Activation('relu', name='block4_act2')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same',
                                                    use_bias=False, name='block4_dwconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_bn2')(x)
    x = layers.MaxPool2D((3,3), strides=(2,2), padding='same', name='block4_pool')(x)

    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block'+str(i+5)

        x = layers.Activation('relu', name=prefix + '_act1')(x)
        x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same',
                                                        use_bias=False, name=prefix + '_dwconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_bn1')(x)
        x = layers.Activation('relu', name=prefix + '_act2')(x)
        x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same',
                                                        use_bias=False, name=prefix + '_dwconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_bn2')(x)
        x = layers.Activation('relu', name=prefix + '_act3')(x)
        x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same',
                                                        use_bias=False, name=prefix + '_dwconv3')(x)
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_bn3')(x)
        x = layers.add([x, residual])

    residual = layers.Conv2D(896, (1,1), strides=(2,2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block13_act1')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3,3), padding='same',
                                                    use_bias=False, name='block13_dwconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_bn1')(x)
    x = layers.Activation('relu', name='block13_act2')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same',
                                                    use_bias=False, name='block13_dwconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_bn2')(x)
    x = layers.MaxPool2D((3,3), strides=(2,2), padding='same', name='block13_pool')(x)
    
    x = layers.add([x, residual])

    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3,3), padding='same',
                                                    use_bias=False, name='block14_dwconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_bn1')(x)
    x = layers.Activation('relu', name='block14_act1')(x)

    x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same',
                                                    use_bias=False, name='block14_dwconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_bn2')(x)
    x = layers.Activation('relu', name='block14_act2')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = tf.keras.Model(inputs, x, name='depthwisenet')

    return model


def xception(input_tensor=None, input_shape=None, classes=2, **kwargs):
    
    
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(img_input)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    #if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    #else:
    #    if pooling == 'avg':
    #        x = layers.GlobalAveragePooling2D()(x)
    #    elif pooling == 'max':
    #        x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = tf.keras.Model(inputs, x, name='xception')

    return model

def nlb_xcnet(input_shape=None, classes=2, trainable=False, **kwargs):

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    pretrain_model = tf.keras.applications.Xception(include_top=False, weights='imagenet')

    inputs = Input(shape=input_shape)

    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3,3), strides=(2,2), use_bias=False, 
                                                    name='block1_dwconv1')(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_bn1')(x)
    x = layers.Activation('relu', name='block1_act1')(x)
    x = layers.DepthwiseConv2D(depth_multiplier=2, kernel_size=(3,3), strides=(2,2), use_bias=False,
                                                    name='block1_dwconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_bn2')(x)
    x = layers.Activation('relu', name='block1_act2')(x)

    x = non_local_block(x, mode="embedded", compression=2)

    x = layers.Conv2D(64, (1,1), padding='same', use_bias=False)(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)
    #print(x.shape)
    #3pretrain_model = delete_layer(pretrain_model.layers[0])

    #for i in range(7):
    #    pretrain_model._layers.pop(0)

    #print(pretrain_model.layers[0].name)
    #pretrain_model.summary()
    #x = pretrain_model(x)
    #pretrain_model.trainable = trainable
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(1024, activation='relu', name='fc_1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(classes, activation='softmax', name='fc_2')(x)

    model = tf.keras.Model(inputs, x, name='nlb_xcnet')
    model.summary()
    plot_model(model, show_shapes=True, to_file='/Users/nikki/Documents/xyu_iterms/aorta_classification/aortaPy/{}.jpg'.format('xception'))

    weights = [layer.get_weights() for layer in pretrain_model.layers[7:]]
    for layer, weight in zip(model.layers[22:-4], weights):
        layer.set_weights(weight)

    return model
    
    


