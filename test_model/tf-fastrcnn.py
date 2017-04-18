########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import numpy as np
import os
from scipy.misc import imread, imresize
from roi_pooling_importer import * # import_roi_pooling_opt 

class_names = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')



# Import roi_pooling_op
roi_pooling_op_dir = os.getenv("HOME") + "/Documents/github/roi_pool/tensorflow-fast-rcnn/tensorflow/core/user_ops/"
roi_pooling_op = import_roi_pooling_op(roi_pooling_op_dir)

class Fast_rcnn:
    def __init__(self, imgs, rois, weights=None, nb_classes=3,
                 roi_pool_output_dim=(7,7), sess=None):
        self.nb_classes = nb_classes
        self.roi_pool_output_dim = roi_pool_output_dim
        self.imgs = imgs
        self.rois = rois
        self.convlayers()
        self.fc_layers()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        # self.pool5 = tf.nn.max_pool(self.conv5_3,
                               # ksize=[1, 2, 2, 1],
                               # strides=[1, 2, 2, 1],
                               # padding='SAME',
                               # name='pool5')

        # roi_pool5
        # First convert NHWC to NCHW
        relu5_transpose = tf.transpose(self.conv5_3, [0, 3, 1, 2])
        output_dim_tensor = tf.constant(self.roi_pool_output_dim)
        roi_pool5, argmax = roi_pooling_op(relu5_transpose, self.rois, output_dim_tensor)

        # ROI pooling outputs in NCRHW.It shouldn't matter,but let's transpose to NRCHW.
        roi_pool5_transpose = tf.transpose(roi_pool5, [0, 2, 1, 3, 4])
        
        # We need to bring this down to 4-d - collapse the ROI and batch together.
        # Should be redundant with next reshape, but whatever
        self.roi_pool5_reshaped = tf.reshape(roi_pool5_transpose, (-1, 512, 7, 7))

    def fc_layers(self):
        # fc6
        with tf.name_scope('fc6') as scope:
            # shape = int(np.prod(self.pool5.get_shape()[1:]))
            shape = int(np.prod(self.roi_pool5_reshaped.get_shape()[1:]))
            fc6w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc6b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            roi_pool5_flat = tf.reshape(self.roi_pool5_reshaped, [-1, shape])
            fc6l = tf.nn.bias_add(tf.matmul(roi_pool5_flat, fc6w), fc6b)
            self.fc6 = tf.nn.relu(fc6l)
            self.parameters += [fc6w, fc6b]

        # fc7
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc7b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            self.fc7 = tf.nn.relu(fc7l)
            self.parameters += [fc7w, fc7b]

        # cls_score
        with tf.name_scope('cls_score') as scope:
            cls_score_w = tf.Variable(tf.truncated_normal([4096, self.nb_classes],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            cls_score_b = tf.Variable(tf.constant(1.0, shape=[self.nb_classes], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.cls_score_l = tf.nn.bias_add(tf.matmul(self.fc7, cls_score_w), cls_score_b)
            self.parameters += [cls_score_w, cls_score_b]

            self.cls_score = tf.nn.softmax(self.cls_score_l)

        # bbox_pred 
        with tf.name_scope('bbox_pred') as scope:
            bbox_pred_w = tf.Variable(tf.truncated_normal([4096, self.nb_classes*4],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            bbox_pred_b = tf.Variable(tf.constant(1.0, shape=[self.nb_classes*4], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.bbox_pred_l = tf.nn.bias_add(tf.matmul(self.fc7, bbox_pred_w), bbox_pred_b)
            self.parameters += [bbox_pred_w, bbox_pred_b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file).item()
        keys = sorted(weights.keys())
        # for i, k in enumerate(keys):
            # print i, k, np.shape(weights[k])
            # sess.run(self.parameters[i].assign(weights[k]))

        
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

if __name__ == '__main__':
    sess = tf.Session()
    # Image placeholder
    imgs = tf.placeholder(tf.float32, [None, None, None, 3])
    # imgs = tf.placeholder(tf.float32, [None, 6000, 1000, 3])

    # ROIs placeholder
    rois_in = tf.placeholder(tf.int32, shape=[None, 4])
    rois = tf.reshape(rois_in, [1, -1, 4])

    nb_cls = len(class_names )
    w = '/home/samy/Documents/mappy/panos/saved_data/vgg16_fast_rcnn_iter_40000.npy'
    # Building Net
    fast_rcnn = Fast_rcnn(imgs, rois, weights=w, nb_classes=nb_cls, sess=sess)

    # img1 = imread('laska.png', mode='RGB')
    # img1 = imresize(img1, (224, 224))

    img1 = imread('1000039898195.jpg', mode='RGB')
    # The width and height of the image
    # Must be divisible by the pooling layers
    img1 = imresize(img1, (1000, 6000))

    # Loading Selective Search
    roi_data = [[(0, 1, 50, 50), (50, 50, 500, 500)]]

    prob = sess.run(fast_rcnn.cls_score, 
                    feed_dict={fast_rcnn.imgs: [img1], fast_rcnn.rois: roi_data})[0]

    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]
