import tensorflow as tf

import os
import shutil
import numpy as np

import DataLoader
import model

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format("aorta"),
    histogram_freq=1, batch_size=,
    write_graph=True, write_grads=False, write_images=True,
    embeddings_freq=0, embeddings_layer_names=None,
    embeddings_metadata=None, embeddings_data=None, update_freq=500
)

