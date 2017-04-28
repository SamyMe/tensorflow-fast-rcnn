from shapely.geometry import box


def iou(ss, gt, threshold=0.5):
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


    return (train_boxes, train_labels)


if __name__=="__main__":

    a = [(0,0,4,4), (3,1,5,6)]
    b = [(1,1,5,5, 1), (3,2,5,7, 2)]
    result = iou(ss=a, gt=b)

    print(result)
