import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import requests
import cv2
import json







with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph("./data/model/model.ckpt.meta")
    saver.restore(sess,"./data/model/model.ckpt")
    print("Medel restore")
    model_path = './data/model/'

    path = './data/model/'

    dir_list = os.listdir(path)
    if len(dir_list) == 0:
        version = 1
    else:
        last_version = len(dir_list)
        version = last_version + 1
    path = path + "{}".format(str(version))
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input_images': tf.saved_model.utils.build_tensor_info(m._new_lr)},
            outputs={'output': tf.saved_model.utils.build_tensor_info(m.logits)},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )
    builder = tf.saved_model.builder.SavedModelBuilder(path)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'generate_images': prediction_signature
        },
        legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op'))
    builder.save(as_text=False)


