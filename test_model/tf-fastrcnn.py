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
roi_pooling_op_dir = os.getenv("HOME") + "/Documents/github/roi_pool/tensorflow-fast-rcnn/tensorflow/core/user_ops/"
roi_pooling_op = import_roi_pooling_op(roi_pooling_op_dir)

class Fast_rcnn:
    def __init__(self, imgs, rois, nb_rois, class_names,
                 roi_pool_output_dim=(7,7), sess=None):

        self.class_names = class_names
        self.nb_classes = len(class_names)
        self.roi_pool_output_dim = roi_pool_output_dim

        # One ROI :D!!!
        self.roi = tf.placeholder(tf.int32, shape=[4])
        self.imgs = imgs
        self.rois = rois
        self.nb_rois = nb_rois
        self.parameters = []
        self.strides = 1

    def build_model(self, weights=None):
        self.convlayers()
        self.fc_layers()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

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
 
        # ##############################################################################################
        # roi_pool5
        # First convert NHWC to NCHW
        relu5_transpose = tf.transpose(self.conv5_3, [0, 3, 1, 2])
        output_dim_tensor = tf.constant(self.roi_pool_output_dim)
        
        # rois = tf.split(self.rois, self.nb_rois, 0)
        # for roi in rois:
        ratio = tf.constant(self.strides)
        self.rois = rois
        self.reshaped_rois = tf.div(rois, ratio)

        roi_pool5, argmax = roi_pooling_op(relu5_transpose, self.reshaped_rois, output_dim_tensor)

        # ROI pooling outputs in NCRHW.It shouldn't matter,but let's transpose to NRCHW.
        roi_pool5_transpose = tf.transpose(roi_pool5, [0, 2, 1, 3, 4])
        
        # We need to bring this down to 4-d - collapse the ROI and batch together.
        # Should be redundant with next reshape, but whatever
        self.roi_pool5_reshaped = tf.reshape(roi_pool5_transpose,
                                            (-1, 512, 
                                            self.roi_pool_output_dim[0], 
                                            self.roi_pool_output_dim[1]))


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
            # self.bbox_pred_l = tf.nn.relu(self.bbox_pred_l)
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


    def fit(self, mode="fc", lr=0.001, nb_iter=100):
        """
        mode :   Option to chose whether to train all the network, the fully connected 
                 part, or only the bbox regression. So it can be either :
                 "all", "fc" or "bbox".
        lr:      Learning Rate.
        nb_iter: Number of iterations.
        """

        var_list = { "fc": self.parameters[-8:],
                     "all": self.parameters,
                     "bbox": self.parameters[-2:],
                    }

        y_box = tf.placeholder(tf.float32, shape=[None, 4*self.nb_classes])
        y_cls = tf.placeholder(tf.float32, shape=[None, self.nb_classes])
        # need to get index of the class like : bbox_pred[cls*4: (cls+1)*4]
        self.bbox_loss = tf.squared_difference(self.bbox_pred_l , y_bbox)
        self.cls_loss = tf.equal(tf.argmax(y_cls,1), tf.argmax(self.cls_score, 1))
        self.cls_loss = tf.reduce_mean(self.cls_loss)

        if mode in ["fc", "all"]:
            self.cost = tf.add(self.bbox_loss, self.cls_loss)
        elif mode=="bbox":
            self.cost = self.bbox_loss 

        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.opt_op = opt.minimize(self.cost, var_list=var_list[mode])


if __name__ == '__main__':
    sess = tf.Session()
    # Image placeholder
    imgs = tf.placeholder(tf.float32, [None ,None, None, 3])
    # imgs = tf.placeholder(tf.float32, [None, 6000, 1000, 3])

    # ROIs placeholder
    rois_in = tf.placeholder(tf.int32, shape=[None, 4])
    rois = tf.reshape(rois_in, [1, -1, 4])

    # Classes names
    class_names = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
    
    # Weights file
    w = '/home/samy/Documents/mappy/panos/saved_data/vgg16_fast_rcnn_iter_40000.npy'

    # Building Net

    # img1 = imread('laska.png', mode='RGB')
    # img1 = imresize(img1, (224, 224))

    # img1 = imread('1000039898195.jpg', mode='RGB')
    # img1 = imread('/home/samy/Pictures/test.jpg', mode='RGB')
    img1 = imread('/home/samy/Pictures/cat.jpg', mode='RGB')
    # img1 = imread('/home/samy/Pictures/cars.jpg', mode='RGB')
    
    # The width and height of the image
    # Must be divisible by the pooling layers
    im_shape = img1.shape
    print(im_shape)
    img1 = imresize(img1, (
                        int(im_shape[0]/16)*16,
                        int(im_shape[1]/16)*16))
    im_shape = img1.shape
    print(im_shape)

    # Loading Selective Search
    # roi_data = [[(0, 1, 50, 50), (20, 20, 100, 100), (50, 50, 100, 50)]]
    roi_data = [[(1, 1, im_shape[0], im_shape[1])]]
    # 15 -> person
    # 7  -> car

    fast_rcnn = Fast_rcnn(imgs, rois, nb_rois=2, class_names=class_names, sess=sess)
    fast_rcnn.build_model(weights=w) 

    prob, bbox = sess.run((fast_rcnn.cls_score, fast_rcnn.bbox_pred_l), 
                    feed_dict={fast_rcnn.imgs: [img1], fast_rcnn.rois: roi_data})

    print(prob)
    print(bbox)
    for i in range(len(prob)):
        prob_ = prob[i]
        bbox_ = bbox[i]

        preds = (np.argsort(prob_)[::-1])[0:5]
        for p in preds:
            print class_names[p], prob_[p]

    # Extracting Boxes
    detect = ['person', 'car', 'cat'] 
    CONF_THRESH = 0.5
    for cls in detect:
        cls_ind = class_names.index(cls)
        cls_boxes = bbox[:, 4*cls_ind:4*(cls_ind+1)]
        cls_scores = prob[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0] 
        for i in keep:
            print(cls, cls_scores[i], cls_boxes)
            draw_shapes(img1, cls_boxes)


