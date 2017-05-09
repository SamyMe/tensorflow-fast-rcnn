import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def import_roi_pooling_op(path_to_op_folder):
    
    # Import the forward op
    roi_pooling_module = tf.load_op_library(path_to_op_folder + "roi_pooling_op.so")
    roi_pooling_op = roi_pooling_module.roi_pooling

    # Import the gradient op
    roi_pooling__grad_module = tf.load_op_library(path_to_op_folder + "roi_pooling_op_grad.so")
    roi_pooling_op_grad = roi_pooling__grad_module.roi_pooling_grad
    
    # Here we register our gradient op as the gradient function for our ROI pooling op. 
    @ops.RegisterGradient("RoiPooling")
    def _roi_pooling_grad(op, grad0, grad1):
        # The input gradients are the gradients with respect to the outputs of the pooling layer
        input_grad = grad0

        # We need the argmax data to compute the gradient connections
        argmax = op.outputs[1]

        # Grab the shape of the inputs to the ROI pooling layer
        input_shape = array_ops.shape(op.inputs[0])

        # Compute the gradient
        backprop_grad = roi_pooling_op_grad(input_grad, argmax, input_shape)

        # Return the gradient for the feature map, but not for the other inputs
        return [backprop_grad, None, None]
    
    return roi_pooling_op
