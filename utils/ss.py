# -*- coding: utf-8 -*-
from blur_project.database import TrainDatabaseHDF5 
from blur_project.database import VectorDatabaseHDF5
from features import SimilarityMask
from selective_search import selective_search
from skimage.transform import resize
import numpy as np


resize_pano = lambda img, resize_factor : (resize(img, output_shape=(img.shape[0]/resize_factor, img.shape[1]/resize_factor, 3)
                     )*255
                     ).astype('uint8')


def save_ss_hdf5(vector_db, ids=None, resize_factor=5, ss_mode="fast", n_jobs=1):

    mul = lambda x : x*resize_factor
    ks = {  "fast": [1000],
            "complete": [100 ,500, 800, 1500, 2500],# 100, 150, 300, 500],
            }

    feature_masks = { "fast" : [ 
                    SimilarityMask(size=1, color=1, 
                                texture=1, fill=1),
                    SimilarityMask(size=1, color=0, 
                                texture=1, fill=1),],
                      "complete" : [ 
                    SimilarityMask(size=0, color=1, 
                                texture=0, fill=0),
                    SimilarityMask(size=0, color=0, 
                                texture=0, fill=1),
                    SimilarityMask(size=1, color=1, 
                                texture=1, fill=1),
                    SimilarityMask(size=1, color=0, 
                                texture=1, fill=1),]}

    color_spaces = {"fast": ['hsv', 'lab', 'rgb'],
                     "complete" : ['hsv', 'lab', 'hue', 'rgb']} #rgi


    if ids==None:
        ids = vector_db.keys()

    errors = []
    i = 0
    for img_id in ids:
        print(img_id)
        i +=1

        # If problem reading, skip
        try:
            img = np.array(vector_db.retrieve_instance(img_id, groups=("Images",))[0])
        except:
            continue

        img = np.rollaxis(img, 0, 3)
        img = resize_pano(img=img, resize_factor=resize_factor)
        try:
            img_ss = selective_search(img, ks=ks[ss_mode], 
                            feature_masks=feature_masks[ss_mode], 
                            color_spaces=color_spaces[ss_mode], n_jobs=n_jobs)
        except:
            try:
                img_ss = selective_search(img, ks=[200, 700, 1500], 
                            feature_masks=feature_masks[ss_mode], 
                            color_spaces=color_spaces[ss_mode], n_jobs=n_jobs)
            except:
                errors.append(img_id)
                continue

        img_ss = np.array([map(mul, (x0, y0, x1, y1)) for score, (y0, x0, y1, x1) in img_ss])
        print("errors : {}".format(errors))
        print("Number of errors: {} / {}".format(len(errors), i))
        vector_db.save_ss(img_id, img_ss, replace=True)
        vector_db.flush()

    print(errors)


if __name__=="__main__":
    hdf5_output_file = '../perf_tests/1400.hdf5'
    vector_db = VectorDatabaseHDF5(output_file=hdf5_output_file)
    save_ss_hdf5(vector_db,ss_mode="complete", resize_factor=8, n_jobs=2)
