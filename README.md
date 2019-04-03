WideResNets & RNNs with weight symmetry
=========

PyTorch 0.3 code for *Exploring Weight Symmetry in Deep Neural Networks*

<https://arxiv.org/abs/1812.11027>

We propose to impose symmetry in neural network parameters to improve parameter usage and make use of dedicated convolution and matrix multiplication routines. Due to significant reduction in the number of parameters as a result of the symmetry constraints, one would expect a dramatic drop in accuracy. Surprisingly, we show that this is not the case, and, depending on network size, symmetry can have little or no negative effect on network accuracy, especially in deep overparameterized networks. We propose several ways to impose local symmetry in recurrent and convolutional neural networks, and show that our symmetry parameterizations satisfy universal approximation property for single hidden layer networks. We extensively evaluate these parameterizations on CIFAR, ImageNet and language modeling datasets, showing significant benefits from the use of symmetry. For instance, our ResNet-101 with channel-wise symmetry has almost 25% less parameters and only 0.2% accuracy loss on ImageNet.


## Main idea in two sentences

We only learn a fraction of weights for a Conv/Linear layer. The other weights of that layer are generated dynamically by repeating the learned weights.


### Requirements

First install [PyTorch](https://pytorch.org), then install [torchnet](https://github.com/pytorch/tnt):

```
pip install git+https://github.com/pytorch/tnt.git@master
```

To train SymmWideResNet on CIFAR10 with triangular symmetry:

```bash
python main.py --width 1 --depth 16 --model resnet --dataset CIFAR10 --symm-type tri
```

To train SymmRNN, please check README.md in symmRNN sub-folder.

## Bibtex

```
@article{hu2018exploring,
  title={Exploring Weight Symmetry in Deep Neural Network},
  author={Hu, Shell Xu and Zagoruyko, Sergey and Komodakis, Nikos},
  journal={arXiv preprint arXiv:1812.11027},
  year={2018}
}
```
