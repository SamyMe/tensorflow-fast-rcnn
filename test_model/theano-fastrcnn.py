# Faster RCNN VGG Net 
"""
The VGG network is adapted from: https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
"""
import theano
import theano.tensor as tensor
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer, RoIPoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer, get_output, get_all_params
from lasagne.init import Normal
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import sgd
import numpy as np
import time

# Testing values
rois = np.asarray([[0., 0., 0., 3., 3.], [0., 0., 0., 7., 7.]], dtype='float32')

# For the SmoothL1Loss defined in Fast RCNN paper, delta = 1
def huber_loss(target, output, delta=1):
	diff = target - output
	ift = 0.5 * (diff ** 2)
	iff = delta * (abs(diff) - delta / 2.)
	return tensor.switch(abs(diff) <= delta, ift, iff).sum()

def build_model(input_var):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    # The values of pooling width and height has been taken from Fast RCNN paper.
    net['pool5'] = RoIPoolLayer(net['conv5_3'], 7, 7,  0.0625, tensor.as_tensor_variable(rois))
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['cls_dense'] = DenseLayer(net['fc7_dropout'], num_units=21, W=Normal(), nonlinearity=None)
    net['cls_score'] = NonlinearityLayer(net['cls_dense'], softmax)
    net['bbox_pred'] = DenseLayer(net['fc7_dropout'], num_units=84, W=Normal(std=0.001), nonlinearity=None)

    return net


def train_net(x, y1, y2, num_iter=10000):
    input_var = tensor.tensor4('input_var')
    cls_target = tensor.ivector('cls_target')
    bbox_target = tensor.ivector('bbox_target')
    network = build_model(input_var)
    cls_score_out, bbox_pred_out = get_output([network['cls_score'], network['bbox_pred']])
    # Computing Loss functions update parameters
    cls_loss = categorical_crossentropy(cls_score_out, cls_target)
    cls_loss = cls_loss.mean()
    bbox_pred_loss = huber_loss(bbox_pred_out, bbox_target)
    bbox_pred_loss = bbox_pred_loss.mean()
    combined_params = get_all_params([network['cls_score'], network['bbox_pred']], trainable=True)
    combined_loss = cls_loss + bbox_pred_loss
    updates = sgd(combined_loss, combined_params, learning_rate=0.001)

    train_net = theano.function([input_var, cls_target, bbox_target], combined_loss, updates=updates)
