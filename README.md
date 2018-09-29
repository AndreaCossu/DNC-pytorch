## DIFFERENTIABLE NEURAL COMPUTER (DNC)

PyTorch (v0.4.1) implementation of Differentiable Neural Computer (DNC).

You can look at the original paper [here.](https://www.nature.com/articles/nature20101)

The implementation provides two different types of controllers:
* LSTM
* MLP (Multi Layer Perceptron)

The task is the Copy task in which a sequence of vectors has to be copied to the output by the DNC.<br>
The  'core' package includes the model implementation by providing Memory and DNC modules. <br>
The 'Controllers' package inside it provides the implementation of both controllers.

The task implementation is provided in the 'CopyTask' package, while the main module to execute it is the 'LaunchCopy' python file, provided in the root folder.<br>
It can be executed by giving one mandatory parameter: the number of epochs of training. <br>
The other parameters are all optional.

The implementation supports batch training, model saving (to resume training or just to test DNC).