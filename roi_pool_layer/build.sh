# Get the path to the Tensorflow headers and whatnot needed for building an op
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

# This should be edited to the path of your CUDA install
CUDA_PATH="/usr/local/cuda-8.0/"

# Build the CUDA kernel
nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op.cu.cc \
-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# Build the C++ forward op
g++ -std=c++11 -shared -o roi_pooling_op.so roi_pooling_op.cc \
roi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L "${CUDA_PATH}lib64/"

# Build the C++ gradient
g++ -std=c++11 -shared -o roi_pooling_op_grad.so roi_pooling_op_grad.cc \
roi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L "${CUDA_PATH}lib64/"
