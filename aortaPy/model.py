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
        self.neck = fpn.FPN(name='fpn')
        self.roi_align = roi_align.PyramidROIAlign(
            pool_shape=self.pool_size,
            name='pyramid_roi_align')
        #self.backbone = resnet.ResNet(depth=101, name='res_net')
        self.classes = classes
        self.epoch = 0
        self.dense0 =layers.Dense(4096, activation='relu', name='fc1')
        self.dense1 =layers.Dense(100, activation='relu', name='fc2')
        #self.xception = tf.keras.applications.Xception(include_top=False, weights='imagenet') 
        #self.inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        if pre_trained == "resnet":
            self.pre_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        elif pre_trained == "vgg":
            self.pre_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        elif pre_trained == "xception":
            self.pre_model = tf.keras.applications.Xception(include_top=False, weights='imagenet')
        elif pre_trained == "inception":
            self.pre_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
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
        img_input = layers.Input(shape=(14, Config.IMAGE_DIM, Config.IMAGE_DIM, 3))
        attr_input = layers.Input(shape=(18,))
        anno_input = layers.Input(shape=(14,))
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
            x = self.pre_model(img_input[:,i,:,:,:])
            C2 = self.pre_model.get_layer('block2_pool').output
            C3 = self.pre_model.get_layer('block3_pool').output
            C4 = self.pre_model.get_layer('block4_pool').output
            C5 = self.pre_model.get_layer('block5_pool').output
            P2, P3, P4, P5, _ = self.neck([C2, C3, C4, C5], 
                                       training=True)
            rcnn_feature_maps = [P2, P3, P4, P5]
            #pooled_rois_list: list of [num_rois, pooled_height, pooled_width, channels].
            #    The width and height are those specific in the pool_shape in the layer
            #    constructor.
            pooled_rois_list = self.roi_align((anno_input[:,i], rcnn_feature_maps), training=True)
            num_pooled_rois_list = [pooled_rois.shape[0] for pooled_rois in pooled_rois_list]
            pooled_rois = tf.concat(pooled_rois_list, axis=0)
        
            y = layers.Conv2D(1024, self.pool_size, activation='relu',
                                padding='valid')(pooled_rois)
            y = layers.Conv2D(1024, (1, 1), activation='relu')(y)
            y = layers.Dense(1024, activation='relu')(y)
            x = layers.Flatten()(x)
            x = layers.Dense(4096, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(1024, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.concatenate([x,y])
            x = layers.Dense(2, activation='sigmoid')(x)
            print(x.shape)
            feature.append(x)
        x = layers.concatenate(feature)
        print(x.shape)
        #x = layers.Dense(23, activation='relu')(x)
        x = layers.Dense(10, activation='relu')(x)
        x = layers.Dense(self.classes, activation='softmax')(x)

        inputs = [img_input, attr_input, anno_input]
        model = tf.keras.Model(inputs, x, name='aorta_model')

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
        log_dir = os.path.join(model_dir, "{:%Y%m%dT%H%M}".format(now))
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
        self.flt = tf.keras.layers.Flatten()
        self.dense0 =tf.keras.layers.Dense(4096, activation='relu', name='fc1')
        self.dense1 =tf.keras.layers.Dense(100, activation='relu', name='fc2')
        #self.xception = tf.keras.applications.Xception(include_top=False, weights='imagenet') 
        #self.inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        if pre_trained == "resnet":
            self.pre_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        elif pre_trained == "vgg":
            self.pre_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        elif pre_trained == "xception":
            self.pre_model = tf.keras.applications.Xception(include_top=False, weights='imagenet')
        elif pre_trained == "inception":
            self.pre_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
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
        img_input = layers.Input(shape=(14, Config.IMAGE_DIM, Config.IMAGE_DIM, 3))
        attr_input = layers.Input(shape=(18,))
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
            x = self.pre_model(img_input[:,i,:,:,:])
            x = layers.Flatten()(x)
            x = layers.Dense(1000, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(100, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(1, activation='sigmoid')(x)
            feature.append(x)
        x = layers.concatenate(feature)
        print(x.shape)
        x = layers.Dense(23, activation='relu')(x)
        x = layers.Dense(10, activation='relu')(x)
        x = layers.Dense(self.classes, activation='softmax')(x)

        inputs = [img_input, attr_input]
        model = tf.keras.Model(inputs, x, name='aorta_model')

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
        log_dir = os.path.join(model_dir, "{:%Y%m%dT%H%M}".format(now))
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
    
    


