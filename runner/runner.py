# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import numpy as np
import os
from scipy.misc import imread, imresize
from utils import * # import_roi_pooling_opt 
# from image_lib import draw_shapes
from models import Fast_rcnn_bigraph as Fast_rcnn


def main():
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
    #img1 = imread('/home/samy/Documents/mappy/panos/saved_data/data/Images/1000041477810.jpg')
    img1 = imread('/home/samy/Pictures/cat.jpg', mode='RGB')
    # img1 = imread('/home/samy/Pictures/cars.jpg', mode='RGB')
    

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
    print(im_shape)

    # Loading Selective Search
    # roi_data = [[(0, 1, 50, 50), (20, 20, 100, 100), (50, 50, 100, 50), (1,1,im_shape[0], im_shape[1])]]
    roi_data = [[(0, 0, im_shape[0], im_shape[1]), (20, 20, 500, 200), (10, 10, im_shape[0], im_shape[1])]]
    # 15 -> person
    # 7  -> car

    fast_rcnn = Fast_rcnn(imgs, rois, nb_rois=2, class_names=class_names, sess=sess)
    fast_rcnn.build_model(weights=w, sess=sess) 
    # fast_rcnn.convolve(img1, sess=sess)

    _ = sess.run((fast_rcnn.save_conv), 
                    feed_dict={fast_rcnn.imgs:[img1]})

    prob, bbox = sess.run((fast_rcnn.cls_score, fast_rcnn.bbox_pred_l), 
                    feed_dict={fast_rcnn.rois: roi_data})
                    # feed_dict={fast_rcnn.rois: roi_data, fast_rcnn.imgs:[img1]})

    # print(prob)
    # print(bbox)
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
            print(cls, cls_scores[i], cls_boxes[i])
            # draw_shapes(img1, cls_boxes)

    # file_writer = tf.summary.FileWriter('..', sess.graph)

if __name__ == '__main__':
    main()
