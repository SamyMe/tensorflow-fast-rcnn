from train_hdf5_2 import TrainDatabaseHDF5 
from iou import iou_flp
from selective_search import selective_search


def gen_train_data(vector_db, train_db, ids=None, resize_factor=5, ss_mode="fast", n_jobs=1):

    if ids==None:
        ids = vector_db.keys()

    for img_id in ids:
        faces, lp, ss = vector_db.retrieve_instance(img_id, groups=("Faces", "LicencePlates", "SelectiveSearch"))
        x_, y_ = iou_flp(ss=ss, faces=faces, lps=lps)
        train_db.save_instance(img=(img_id, img), x=x_, y=y_)

    vector_db.flush()
