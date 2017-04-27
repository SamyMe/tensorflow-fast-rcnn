# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import numpy as np
import os
from scipy.misc import imread, imresize
from roi_pooling_importer import * # import_roi_pooling_opt 
from image_lib import draw_shapes


# Import roi_pooling_op
roi_pooling_op_dir = os.getenv("HOME") + "/Documents/git/tensorflow/bazel-bin/tensorflow/core/user_ops/"
# roi_pooling_op_dir = os.getenv("HOME") + "/Documents/git/roi_pool/tensorflow-fast-rcnn/tensorflow/core/user_ops/"
# roi_pooling_op_dir = '../lib/layers/'
roi_pooling_op = import_roi_pooling_op(roi_pooling_op_dir)

class Fast_rcnn:
    def __init__(self, imgs, rois, nb_img, nb_rois, class_names,
                 roi_pool_output_dim=(7,7), sess=None):

        self.class_names = class_names
        self.nb_classes = len(class_names)
        self.roi_pool_output_dim = roi_pool_output_dim

        # One ROI :D!!!
        self.roi = tf.placeholder(tf.int32, shape=[4])
        self.imgs = imgs
        self.rois = rois
        self.nb_rois = nb_rois
        self.nb_img = nb_img
        self.parameters = []
        self.strides = 1

    def build_model(self, weights=None, sess=None):
        self.conv_saved = tf.Variable(np.zeros([1, 32, 32, 512]), dtype=tf.float32, name="saved_conv")
        # sess.run(tf.variables_initializer([self.conv_saved]))
        self.convlayers()
        self.def_roi_fc_params()
        self.def_rois()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

        self.roi_fc_shape = (4096, 25088)

    def conv_2d(self, scope_name, input, kernel_size, stride, padding='SAME'):
        """
        scope_name : 1_1
        """
        self.strides = self.strides * stride
        with tf.name_scope('conv'+scope_name) as scope:
            kernel = tf.Variable(tf.truncated_normal(kernel_size, dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding=padding)
            biases = tf.Variable(tf.constant(0.0, shape=[kernel_size[3]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

            return conv

    def max_pool(self, input, name, kernel_size, stride, padding='SAME'):
        self.strides = self.strides * stride
        pool = tf.nn.max_pool(input,
                               ksize=kernel_size,
                               strides=[1, stride, stride, 1],
                               padding='SAME',
                               name=name)
        return pool


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean


        # conv1_1
        self.conv1_1 = self.conv_2d('1_1', input=images, kernel_size=[3, 3, 3, 64], stride=1) 

        # conv1_2
        self.conv1_2 = self.conv_2d('1_2', input=self.conv1_1, kernel_size=[3, 3, 64, 64], stride=1) 

        # pool1
        self.pool1 = self.max_pool(input=self.conv1_2, name='pool1', kernel_size=[1, 2, 2, 1], stride=2)

        # conv2_1
        self.conv2_1 = self.conv_2d('2_1', input=self.pool1, kernel_size=[3, 3, 64, 128], stride=1) 

        # conv2_2
        self.conv2_2 = self.conv_2d('2_2', input=self.conv2_1, kernel_size=[3, 3, 128, 128], stride=1) 

        # pool2
        self.pool2 = self.max_pool(input=self.conv2_2, name='pool2', kernel_size=[1, 2, 2, 1], stride=2)

        # conv3_1
        self.conv3_1 = self.conv_2d('3_1', input=self.pool2, kernel_size=[3, 3, 128, 256], stride=1) 

        # conv3_2
        self.conv3_2 = self.conv_2d('3_2', input=self.conv3_1, kernel_size=[3, 3, 256, 256], stride=1) 

        # conv3_3
        self.conv3_3 = self.conv_2d('3_3', input=self.conv3_2, kernel_size=[3, 3, 256, 256], stride=1) 

        # pool3
        self.pool3 = self.max_pool(input=self.conv3_3, name='pool3', kernel_size=[1, 2, 2, 1], stride=2)

        # conv4_1
        self.conv4_1 = self.conv_2d('4_1', input=self.pool3, kernel_size=[3, 3, 256, 512], stride=1) 

        # conv4_2
        self.conv4_2 = self.conv_2d('4_2', input=self.conv4_1, kernel_size=[3, 3, 512, 512], stride=1) 

        # conv4_3
        self.conv4_3 = self.conv_2d('4_3', input=self.conv4_2, kernel_size=[3, 3, 512, 512], stride=1) 

        # pool4
        self.pool4 = self.max_pool(input=self.conv4_3, name='pool4', kernel_size=[1, 2, 2, 1], stride=2)

        # conv5_1
        self.conv5_1 = self.conv_2d('5_1', input=self.pool4, kernel_size=[3, 3, 512, 512], stride=1) 

        # conv5_2
        self.conv5_2 = self.conv_2d('5_2', input=self.conv5_1, kernel_size=[3, 3, 512, 512], stride=1) 

        # conv5_3
        self.conv5_3 = self.conv_2d('5_3', input=self.conv5_2, kernel_size=[3, 3, 512, 512], stride=1) 


        self.save_conv = tf.assign(ref=self.conv_saved, value=self.conv5_3, validate_shape=False)

        # pool5
        # self.pool5 = self.max_pool(input=self.conv5_3,
                               # kernel_size=[1, 2, 2, 1],
                               # stride=2,
                               # name='pool5')

        # # roi_pool5
        # # First convert NHWC to NCHW
        # relu5_transpose = tf.transpose(self.conv1_1, [0, 3, 1, 2])
        # output_dim_tensor = tf.constant((104,104))
        # 
        # # rois = tf.split(self.rois, self.nb_rois, 0)
        # # for roi in rois:
        # ratio = tf.constant(1)
        # self.rois = rois
        # self.reshaped_rois = tf.div(rois, ratio)
 
        # roi_pool5, argmax = roi_pooling_op(relu5_transpose, self.reshaped_rois, output_dim_tensor)
 
        # # ROI pooling outputs in NCRHW.It shouldn't matter,but let's transpose to NRCHW.
        # roi_pool5_transpose = tf.transpose(roi_pool5, [0, 2, 1, 3, 4])
        # 
        # # We need to bring this down to 4-d - collapse the ROI and batch together.
        # # Should be redundant with next reshape, but whatever
        # self.roi_pool5_reshaped2 = tf.reshape(roi_pool5_transpose, (-1, 64, 104, 104))
 
        # ###############################################################################


    def def_rois(self):
        self.out_bbox = []
        self.out_cls = []

        self.img_list = tf.split(value=self.rois, num_or_size_splits=self.nb_img, axis=0)

        for img in self.img_list:
            rois_list = tf.split(value=img, num_or_size_splits=self.nb_rois, axis=1)
            img_bbox = []
            img_cls = []
            for roi in rois_list:
                c, b = self.roi_fc_layers(roi)
                img_bbox.append(b)
                img_cls.append(c)

            self.out_bbox.append(img_bbox)
            self.out_cls.append(img_cls)

    def def_roi_fc_params(self):

        # FC6
        with tf.name_scope('fc6') as scope:
            # shape = int(np.prod(self.roi_pool5_reshaped.get_shape()[1:]))
            shape = 25088
            fc6w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc6b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [fc6w, fc6b]

        # FC7
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')

            fc7b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [fc7w, fc7b]

        # CLS_SCORE
        with tf.name_scope('cls_score') as scope:
            cls_score_w = tf.Variable(tf.truncated_normal([4096, self.nb_classes],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            cls_score_b = tf.Variable(tf.constant(1.0, shape=[self.nb_classes], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [cls_score_w, cls_score_b]

        # BBOX_PRED 
        with tf.name_scope('bbox_pred') as scope:
            bbox_pred_w = tf.Variable(tf.truncated_normal([4096, self.nb_classes*4],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            bbox_pred_b = tf.Variable(tf.constant(1.0, shape=[self.nb_classes*4], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [bbox_pred_w, bbox_pred_b]


    def roi_fc_layers(self, roi_placeholder):
        # Loading shared weights
        (fc6w, fc6b,
        fc7w, fc7b,
        cls_score_w, cls_score_b,
        bbox_pred_w, bbox_pred_b) = self.parameters[-8:]

        # roi_pool5
        # First convert NHWC to NCHW
        relu5_transpose = tf.transpose(self.conv5_3, [0, 3, 1, 2])
        # relu5_transpose = tf.transpose(self.conv_saved, [0, 3, 1, 2])
        output_dim_tensor = tf.constant(self.roi_pool_output_dim)
        
        # rois = tf.split(self.rois, self.nb_rois, 0)
        # for roi in rois:
        ratio = tf.constant(self.strides)
        rois = self.rois
        self.reshaped_rois = tf.div(rois, ratio)

        roi_pool5, argmax = roi_pooling_op(relu5_transpose, roi_placeholder, output_dim_tensor)

        # ROI pooling outputs in NCRHW.It shouldn't matter,but let's transpose to NRCHW.
        roi_pool5_transpose = tf.transpose(roi_pool5, [0, 2, 1, 3, 4])
        
        # We need to bring this down to 4-d - collapse the ROI and batch together.
        # Should be redundant with next reshape, but whatever
        roi_pool5_reshaped = tf.reshape(roi_pool5_transpose,
                                            (-1, 512, 
                                            self.roi_pool_output_dim[0], 
                                            self.roi_pool_output_dim[1]))

        # fc6
        with tf.name_scope('fc6') as scope:
            # shape = int(np.prod(pool5.get_shape()[1:]))
            shape = int(np.prod(roi_pool5_reshaped.get_shape()[1:]))
            shape = 25088
            roi_pool5_flat = tf.reshape(roi_pool5_reshaped, [-1, shape])
            fc6l = tf.nn.bias_add(tf.matmul(roi_pool5_flat, fc6w), fc6b)
            fc6 = tf.nn.relu(fc6l)

        # fc7
        with tf.name_scope('fc7') as scope:
            fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
            fc7 = tf.nn.relu(fc7l)

        # cls_score
        with tf.name_scope('cls_score') as scope:
            cls_score_l = tf.nn.bias_add(tf.matmul(fc7, cls_score_w), cls_score_b)
            cls_score = tf.nn.softmax(cls_score_l)

        # bbox_pred 
        with tf.name_scope('bbox_pred') as scope:
            bbox_pred_l = tf.nn.bias_add(tf.matmul(fc7, bbox_pred_w), bbox_pred_b)
            # self.bbox_pred_l = tf.nn.relu(self.bbox_pred_l)

        return cls_score, bbox_pred_l


    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file).item()
        keys = sorted(weights.keys())

        # ['bbox_pred',
        # 'cls_score',
        # 'conv1_1',
        # 'conv1_2',
        # 'conv2_1',
        # 'conv2_2',
        # 'conv3_1',
        # 'conv3_2',
        # 'conv3_3',
        # 'conv4_1',
        # 'conv4_2',
        # 'conv4_3',
        # 'conv5_1',
        # 'conv5_2',
        # 'conv5_3',
        # 'fc6',
        # 'fc7']
        
        wb = weights['bbox_pred']
        sess.run(self.parameters[-1].assign(wb[1].T))
        sess.run(self.parameters[-2].assign(wb[0].T))

        wb = weights['cls_score']
        sess.run(self.parameters[-3].assign(wb[1].T))
        sess.run(self.parameters[-4].assign(wb[0].T))

        i = 0
        for k in keys[2:]:
            print k, np.shape(weights[k][0]), np.shape(weights[k][1])
            wb = weights[k]
            sess.run(self.parameters[i].assign(wb[0].T))
            sess.run(self.parameters[i+1].assign(wb[1].T))
            i += 2

