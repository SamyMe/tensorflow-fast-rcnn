# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import numpy as np
import os
from scipy.misc import imread, imresize
from roi_pooling_importer import * # import_roi_pooling_opt 
from image_lib import draw_shapes
from uni_graph import Fast_rcnn


def main():
    sess = tf.Session()
    # Image placeholder
    imgs_ph = tf.placeholder(tf.float32, [None ,None, None, 3])
    # imgs = tf.placeholder(tf.float32, [None, 6000, 1000, 3])

    # ROIs placeholder
    rois_ph = tf.placeholder(tf.int32, shape=[None, None, 4])

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
    # img1 = imread('/home/blur/Documents/data/2000010000008.jpg', mode='RGB')
    

    # ss_file = '/home/samy/Documents/mappy/panos/ss.pkl'

    # The width and height of the image
    # Must be divisible by the pooling layers
    im_shape = img1.shape
    print(im_shape)
    img1 = imresize(img1, (
                        int((im_shape[0]/(16))*16),
                        int((im_shape[1]/(16))*16)))
                        # int(((im_shape[0]/4)/16)*16),
                        # int((im_shape[1]/4)/16)*16))
    im_shape = img1.shape
    print(im_shape)

    # Loading Selective Search
    # roi_data = [[(0, 1, 50, 50), (20, 20, 100, 100), (50, 50, 100, 50), (1,1,im_shape[0], im_shape[1])]]
    roi_data = np.array([
                [(1,1,2,2), (1, 1, im_shape[0]-10, im_shape[1]-10), (1, 1, im_shape[1]-10, im_shape[0]-10), (20, 20, 500, 200)],
                # [(1,1,2,2),(1, 1, im_shape[0]-10, im_shape[1]-10), (1, 1, im_shape[1]-10, im_shape[0]-10), (20, 20, 500, 200)]
                ])

    # roi_data_all = np.load('/home/blur/Documents/data/ss.npy')
    # print(roi_data)
    # 15 -> person
    # 7  -> car

    fast_rcnn = Fast_rcnn(imgs_ph, rois_ph, nb_img=1, nb_rois=4, class_names=class_names, sess=sess)
    fast_rcnn.build_model(weights=w, sess=sess) 
    # Generate graph
    # file_writer = tf.summary.FileWriter('../train', sess.graph)
    # fast_rcnn.convolve(img1, sess=sess)

    # split_list = sess.run(fast_rcnn.rois_list,
                    # feed_dict={fast_rcnn.imgs:[img1],
                               # fast_rcnn.rois: roi_data})
 
    # print(split_list)

    w = 0
    # for w in range(200):
        # roi_data = [roi_data_all[w*50:(w+1)*50]]

    file_writer = tf.summary.FileWriter('..', sess.graph)
    prob, bbox  = sess.run((fast_rcnn.out_cls, fast_rcnn.out_bbox),
                    feed_dict={fast_rcnn.imgs:[img1],
                               fast_rcnn.rois: roi_data})


    # prob = prob[0]
    # bbox = bbox[0]
    bbox = np.stack(bbox, 1)
    print(bbox.shape)
    prob = np.stack(prob, 1)
    print(prob.shape)
    print("!!! PROBS !!!")
    # prob = np.array(prob)
    # shape = prob.shape
    # prob = prob.reshape(shape[0], shape[1], shape[3])
    # bbox = np.array(bbox)
    # shape = bbox.shape
    # bbox = bbox.reshape(shape[0], shape[1], shape[3])

    #######################
    ## Best class per Bbox
    #######################

    # for img in range(len(prob)):
        # # For each image
        # print()
        # print("IMAGE {}".format(img))
 
        # for i in range(len(prob[img])):
            # # For each roi
            # print("BBOX {}".format(i))
            # 
            # prob_ = prob[img][i]
            # bbox_ = bbox[img][i]
 
            # preds = np.argsort(prob_)[::-1][0:2]
            # for p in preds:
                # print class_names[p], prob_[p]

    # Extracting Boxes
    detect = ['person', 'car', 'cat'] 
    CONF_THRESH = 0.4
    for i in range(len(roi_data)):

        print("Image {}".format(i))
        for cls in detect:
            cls_ind = class_names.index(cls)
            cls_boxes = bbox[i][:, 4*cls_ind:4*(cls_ind+1)]
            cls_scores = prob[i][:, cls_ind]
            keep = np.where(cls_scores >= CONF_THRESH)[0] 
            for j in keep:
                print(cls, cls_scores[j], cls_boxes[j], roi_data[i][j])
                print(rescale(roi_data[i][j], cls_boxes[i], im_shape=img1.shape))
                draw_shapes(img1, [rescale(roi_data[i][j], cls_boxes[i], im_shape=img1.shape)], img_name="image_{}_{}_{}.jpg".format(i,j,w))

        # file_writer = tf.summary.FileWriter('..', sess.graph)

def rescale(roi, t, im_shape):
    # Add crop in reconstruction !
    w = roi[3]
    h = roi[2]
    ctr_x = roi[1] + 0.5*w
    ctr_y = roi[0] + 0.5*h

    dx = t[0]
    dy = t[1]
    dw = t[2]
    dy = t[3]

    pred_w = np.exp(dw)*w
    pred_h = np.exp(dy)*h
    pred_ctr_x = dx*w + ctr_x
    pred_ctr_y = dy*h + ctr_y

    pred_x = max(pred_ctr_x - 0.5*pred_w, 0)
    pred_y = max(pred_ctr_y - 0.5*pred_h, 0)

    return (pred_x,
            pred_y, 
            min(pred_w, im_shape[1]-pred_x-2), 
            min(pred_h, im_shape[0]-pred_y-2))


if __name__ == '__main__':
    main()
