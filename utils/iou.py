from shapely.geometry import box


def iou(gt, ss, threshold=0.5):
    train_boxes = []
    for bbox in ss:
        for gt_box in gt:
            x1, y1, x2, y2 = bbox
            # box(minx, miny, maxx, maxy, ccw=True)
            ss_box = box(x1, y1, x2, y2)

            x3, y3, x4, y4, cls = gt
            gt_box = (x3, y3, x4, y4)

            u = ss_box.union(gt_box).area
            i = ss_box.intersection(gt_box).area
            
            if i/o > threshol:
                ss_box.append(cls)
                train_box.append(ss_box)


    return train_boxes
