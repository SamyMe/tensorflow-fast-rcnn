## Installation :
Complete Installation guide for building Tensorflow with GPU support from source with custom Ops.

### CUDA/CuDNN:

- Install requirements: 
```
sudo apt-get install build-essential linux-headers-$(uname -r)
```
- Download CUDA from [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
- Install
```sudo dpkg -i FILE.deb
sudo apt-get update
sudo apt-get install cuda
```
- Download CuDNN from [NVIDIA website](https://developer.nvidia.com/cudnn)
- Extract file and copy files
``` cp include/* /usr/local/cuda-8.0/include/```
cp lib64/* /usr/local/cuda-8.0/lib64/```
- Restart computer
- Verify installation 
```nvidia-smi```

Fore more info check the following links :
- Nvidia [pdf guide](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf)
- Nvidia [CUDA Doc](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)
- Nvidia [CUDA Doc](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)

### Bazel:

ADD Source:
```echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```

Install 
```sudo apt-get update && sudo apt-get install bazel
```

### GCC 4.9:
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-4.9
```

### Tensorflow(from source) + roi\_pooling

- Install requirements:
```
sudo apt-get install libcupti-dev
sudo apt-get install python-numpy python-dev python-pip python-wheel #For python2
sudo pip install six numpy wheel 
```
- Clone project:
```
git clone https://github.com/tensorflow/tensorflow 
```
- Get disired branch (1.0 in our case)
```
cd tensorflow
git checkout r1.0
```
- Copy roi\_pool files to tensorflow source (let TFPATH be tensorflow repo path)
```
cp roi_pool_layer/ $TFPATH/tensorflow/tensorflow/core/user_ops/
```

- Configure and specify */usr/bin/gcc-4.9* as gcc version to be used
```
./configure
```
- Compile Tensorflow with gpu support
```
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
```

- Generate pip installation
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

- Install with pip:
```
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.0.1-py2-none-any.whl
```

- Check if Tensorflow detects GPU:
```
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

- Compile roi op:
```
bazel build --config opt //tensorflow/core/user_ops:roi_pooling_op.so
bazel build --config opt //tensorflow/core/user_ops:roi_pooling_op_grad.so
```

- Test if works:
```
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
#Folder containing .so files
path_to_op_folder = $TFPATH/tensorflow/bazel-bin/tensorflow/core/user_ops/
#Import the forward op
roi_pooling_module = tf.load_op_library(path_to_op_folder + "roi_pooling_op.so")
roi_pooling_op = roi_pooling_module.roi_pooling
#Import the gradient op
roi_pooling__grad_module = tf.load_op_library(path_to_op_folder + "roi_pooling_op_grad.so")
roi_pooling_op_grad = roi_pooling__grad_module.roi_pooling_grad
```

- Install additional requirements:
```pip install -r requirements.txt```

For more info about TF installation [Official doc](https://www.tensorflow.org/install/install_sources).
More info about Adding a New Op [Official doc] (https://www.tensorflow.org/extend/adding_an_op#compile_the_op_using_bazel_tensorflow_source_installation)
Roi original c++ implementation [src](https://github.com/zplizzi/tensorflow-fast-rcnn).
