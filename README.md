# Fine-Grained Vehicle Classification

This repository contains code for the final project of Stanford's CS230 (Deep Learning) on fine-grained vehicle classification, applying transfer learning and the VGG16 CNN architecture to the Stanford Cars dataset in order to classify 196 classes of vechiles (make, model, and year).

## Dataset

The Stanford Cars dataset can be found [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

### Required Packages/Software
* Python >= 3.6
* numpy
* Tensorflow-gpu 1.12.0
* Keras 2.2.4
* matplotlib
* scikit-learn
* PIL
* tqdm
* cv2

### Running Code
Once the dataset has been downloaded, required packages have been installed, and VGG16 model weights have been loaded from [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5), run python baseline_model.py or python train_vgg_model.py to train a model.

## Authors

* **Nicholas Benavides**
* **Christian Tae**

## Acknowledgments

* Thanks to [Xiaotian-WANG](https://github.com/Xiaotian-WANG/Fine-Tune-VGG-Networks-Based-on-Stanford-Cars) for code used in the transfer learning model.