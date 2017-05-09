# -*- coding: utf-8 -*-

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import numpy as np
import os
from scipy.misc import imread, imresize
from roi_pooling_importer import * # import_roi_pooling_opt 
from image_lib import draw_shapes
from bi_graph import Fast_rcnn


def fit(net, x_train, y_train, mode="fc", lr=0.001, nb_iter=100):
    """
    mode :   Option to chose whether to train all the network, the fully connected 
             part, or only the bbox regression. So it can be either :
             "all", "fc" or "bbox".
    lr:      Learning Rate.
    nb_iter: Number of iterations.
    """


    var_list = { "fc": net.parameters[-8:],
                 "all": net.parameters,
                 "bbox": net.parameters[-2:],
                }

    y_box = tf.placeholder(tf.float32, shape=[None, 4], name="y_box")
    y_cls = tf.placeholder(tf.int64, shape=[None, 1], name="y_cls")
    box_ind = tf.placeholder(tf.int64, shape=[None, 2], name="box_ind")
    # need to get index of the class like : bbox_pred[cls*4: (cls+1)*4]

    pred_cls = tf.argmax(net.cls_score, 1)

    pred_bbox_reshaped = tf.reshape(net.bbox_pred_l, [-1, 21, 4])
    pred_bbox_slice = tf.gather_nd(pred_bbox_reshaped, box_ind)

    bbox_loss = tf.squared_difference(pred_bbox_slice , y_box)
    bbox_loss = tf.reduce_mean(bbox_loss)

    cls_loss = tf.to_float(tf.equal(y_cls, pred_cls))
    cls_loss = tf.reduce_mean(cls_loss)

    if mode in ["fc", "all"]:
        cost = tf.add(bbox_loss, cls_loss)
    elif mode=="bbox":
        cost = bbox_loss 

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    opt_op = opt.minimize(cost, var_list=var_list[mode])

    # ___ RUN ___ #

    inds = []
    nb_example = y_train.shape[0]
    for i in range(nb_example):
        cls = y_train[i,-1]
        inds.append([i, cls])

    inds = np.array(inds)

    for i in range(nb_iter):
        loss, _ = sess.run((cost, opt_op), feed_dict={ net.rois:x_train, 
                                            box_ind:inds,
                                            y_cls:y_train[:,-1:].astype(np.int64), 
                                            y_box:y_train[:,:-1]})
 
        print('iteration {} : loss {}'.format(i, loss))


if __name__=="__main__":
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
    w = '/home/blur/Documents/saved_data/vgg16_fast_rcnn_iter_40000.npy'

    # Building Net

    # img1 = imread('laska.png', mode='RGB')
    # img1 = imresize(img1, (224, 224))

    # img1 = imread('1000039898195.jpg', mode='RGB')
    img1 = imread('/home/blur/Pictures/cat.jpg', mode='RGB')
    

    # ss_file = '/home/samy/Documents/mappy/panos/ss.pkl'

    # The width and height of the image
    # Must be divisible by the pooling layers
    im_shape = img1.shape
    print(im_shape)
    img1 = imresize(img1, (
                        int((im_shape[0]/16)*16),
                        int((im_shape[1]/16)*16)))
                        # int(((im_shape[0]/4)/16)*16),
                        # int((im_shape[1]/4)/16)*16))
    im_shape = img1.shape

    fast_rcnn = Fast_rcnn(imgs, rois, nb_rois=2, class_names=class_names, sess=sess)
    fast_rcnn.build_model(weights=w, sess=sess) 
    # fast_rcnn.convolve(img1, sess=sess)

    _ = sess.run((fast_rcnn.save_conv), 
                    feed_dict={fast_rcnn.imgs:[img1]})

    # prob, bbox = sess.run((fast_rcnn.cls_score, fast_rcnn.bbox_pred_l), 
                    # feed_dict={fast_rcnn.rois: roi_data})

    x_train = np.array([[[0,0,100,100], [1,1,60,60]]])
    y_train = np.array([[1,1,80,80,1], [2,2,50,50,2]])
    # Loading Selective Search
    fit(net=fast_rcnn, x_train=x_train, y_train=y_train, mode="fc", 
            lr=0.001, nb_iter=10)
