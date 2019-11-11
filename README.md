# Hessian-based-analysis-tensorflow
## Description
This is the implementation of an additional experiment with tensorflow in the following paper.

paper: https://arxiv.org/pdf/1802.08241.pdf

Implementation by the author: https://github.com/amirgholami/HessianFlow

I will upload the slide deck of DLHacks later.

## Docker version
- Docker version 19.03.2
- docker-compose version 1.24.1

## Experiment
In the paper, they use the Hessian matrix w.r.t input, but this repository uses the Hessian matrix w.r.t weight parameters for analysis. As in the paper, I conducted the experiment using CIFAR-10 for image classification. The model architecture is as follows.

|Dataset|Model architecture|
|:---|:---|
|CIFAR-10|Conv(3,3,64) - Conv(3,3,64) - MaxPool(2,2) - Conv(3,3,128) - Conv(3,3,128) - MaxPool(2,2) - Dense(256) - Dense(256) - Softmax(10)|

## how to use
```
$ docker-compose build
$ UID=${UID} GID=${GID} docker-compose run --rm app /bin/bash
(docker_container)$ python experiment_cifar10_sgd.py
```

## Results
#### The top-20 eigenvalues of the Hessian matrix for the CIFAR-10 dataset at the points of convergence reached by Momentum SGD.

<img src="https://github.com/mtkwT/Hessian-based-analysis-tensorflow/blob/master/notebooks/hessian_spectral.png" width="550" height="420">

#### The local training loss landscapes at the points of convergence reached by Momentum SGD.

<table border="0">
<tr>
<td><img src="https://github.com/mtkwT/Hessian-based-analysis-tensorflow/blob/master/notebooks/cifar10_sgd_train_loss_surface_batchsize128.png" width="350" height="240"></td>
<td><img src="https://github.com/mtkwT/Hessian-based-analysis-tensorflow/blob/master/notebooks/cifar10_sgd_train_loss_surface_batchsize256.png" width="350" height="240"></td>
</tr>
</table>
<table border="0">
<tr>
<td><img src="https://github.com/mtkwT/Hessian-based-analysis-tensorflow/blob/master/notebooks/cifar10_sgd_train_loss_surface_batchsize512.png" width="350" height="240"></td>
<td><img src="https://github.com/mtkwT/Hessian-based-analysis-tensorflow/blob/master/notebooks/cifar10_sgd_train_loss_surface_batchsize1024.png" width="350" height="240"></td>
</tr>
