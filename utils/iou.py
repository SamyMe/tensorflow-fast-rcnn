from shapely.geometry import box
from random import random
import numpy as np
from .image_lib import draw_shapes


def iou(ss, gt, threshold=0.5, bg_threshold=0.2):
    train_boxes = []
    train_labels = []
    for gt_bbox in gt:
        x3, y3, x4, y4, cls = gt_bbox
        gt_box = box(x3, y3, x4, y4)
        train_boxes.append(gt_bbox[:-1])
        train_labels.append(gt_bbox)

        for bbox in ss:
            x1, y1, x2, y2 = bbox
            # box(minx, miny, maxx, maxy, ccw=True)
            ss_box = box(x1, y1, x2, y2)

            u = ss_box.union(gt_box).area
            i = ss_box.intersection(gt_box).area
            
            if i/u > threshold:
                train_boxes.append(bbox)
                train_labels.append(gt_bbox)

            elif random()<bg_threshold:
                train_boxes.append(bbox)
                train_labels.append(list(bbox)+[0,])

    return (train_boxes, train_labels)


def get_label(gt, bbox):
    return [(gt[1]-bbox[1])/bbox[3], (gt[0]-bbox[0])/bbox[2], 
            np.log(gt[3]/bbox[3]), np.log(gt[2]/bbox[2])]


def iou(ss, gt, threshold=0.5, bg_threshold=0.1):
    train_boxes = []
    train_labels = []
    for gt_bbox in gt:
        y, x, h, w, cls = gt_bbox
        gt_box = box(x, y, x+w, y+h)
        train_boxes.append(gt_bbox)
        train_labels.append([0, 0, 0, 0, cls])

        for bbox in ss:
            y, x, h, w = bbox
            # box(minx, miny, maxx, maxy, ccw=True)
            ss_box = box(x, y, x+w, y+h)

            u = ss_box.union(gt_box).area
            i = ss_box.intersection(gt_box).area
            
            if i/u > threshold:
                train_boxes.append(bbox)
                label = get_label(gt_bbox, bbox)
                train_labels.append(label+[cls,])

            # Randomly adding some bg class bbox
            elif random()<bg_threshold:
                train_boxes.append(bbox)
                train_labels.append([0, 0, 0, 0, 0])

    return (train_boxes, train_labels)


def iou_flp(img, ss, faces, lps, threshold=0.1, 
            bg_threshold=0.001, max_by_gt=200, debug=False):
    nb_labels = 0
    nb_bg = 0

    train_boxes = []
    train_labels = []

    for gt, cls in [(faces, 1), (lps, 2)]:

        for gt_bbox in gt:

            example = 0

            # y, x, h, w = gt_bbox

            ## TEMPORARY ##
            x, y, w, h = gt_bbox
            # x, y, x1, y1 = gt_bbox
            # w = x1 - x 
            # h = y1 - y
            # gt_bbox = y, x, h, w
            gt_x, gt_y, gt_w, gt_h = x, y, w, h 
            ###############

            gt_box = box(x, y, x+w, y+h)
            train_boxes.append(gt_bbox)
            train_labels.append([0, 0, 0, 0, cls])

            for bbox in ss:

                # y, x, h, w = bbox
                ## TEMPORARY ##
                x, y, x1, y1 = bbox
                w = x1 - x 
                h = y1 - y
                bbox = y, x, h, w
                ###############

                # box(minx, miny, maxx, maxy, ccw=True)
                ss_box = box(x, y, x+w, y+h)

                u = ss_box.union(gt_box).area
                i = ss_box.intersection(gt_box).area
                
                if i/u > threshold:
                    if debug==True and random()<0.5:
                        print("{} / {}".format(i,u))
                        print("ss: {}, {}, {}, {} ".format(x, y, w, h)) 
                        print("gt: {}, {}, {}, {}".format(gt_x, gt_y, gt_w, gt_h))
                        draw_shapes(img, faces=[(x, y, w, h),
                                                (gt_x, gt_y, gt_w, gt_h)], lp=[])

                    train_boxes.append(bbox)
                    label = get_label(gt_bbox, bbox)
                    nb_labels += 1
                    train_labels.append(label+[cls,])
                    
                    example += 1
                    if example>max_by_gt :
                        break


                # Randomly adding some bg class bbox
                elif i/u==0 and nb_bg<max_by_gt:
                    train_boxes.append(bbox)
                    train_labels.append([0, 0, 0, 0, 0])
                    nb_bg += 1

    print("   * Labels : {} | BG : {}".format(nb_labels, nb_bg))
    return (train_boxes, train_labels)


if __name__=="__main__":

    a = [(0,0,4,4), (3,1,5,6)]
    b = [(1,1,5,5, 1), (3,2,5,7, 2)]
    result = iou(ss=a, gt=b)

    print(result)

    b = [(1,1,5,5)]
    c = [(3,2,5,7)]
    result = iou_flp(ss=a, faces=b, lps=c)

    print(result)
