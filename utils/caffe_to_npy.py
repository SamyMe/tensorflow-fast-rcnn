import sys
import os.path as osp
import numpy as np

fast_rcnn_dir = sys.argv[0]

# caffe path
to_append = osp.join(fast_rcnn_dir, 'caffe-fast-rcnn', 'python')
sys.path.append(to_append)

# roi_pooling path
to_append = osp.join(fast_rcnn_dir, 'lib')
sys.path.append(to_append)

import caffe

prototxt = sys.argv[1]
caffemodel = sys.argv[2]
output_file = sys.argv[3]

# Exemple:
# net = caffe.Net('models/VGG16/train.prototxt', 'data/fast_rcnn_models/vgg16_fast_rcnn_iter_40000.caffemodel', caffe.TEST)

# Reading blub
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

# Layer names
keys = net.params.keys()

# Weights as a dictionary
weights = dict([(key, (net.params[key][0].data, net.params[key][1].data)) for key in keys])

# Saving
np.save(output_file, weights)
