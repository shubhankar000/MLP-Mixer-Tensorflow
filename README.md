# MLP-Mixer-Tensorflow
Layer-based tensorflow implementation of MLP-Mixer (https://arxiv.org/abs/2105.01601)

### Layers
3 Layers have been created to access via the standard tensorflow API. The layers are
- CreatePatches
- PerPatchFullyConnected
- MLPMixer

More information and the implementation of these layers can be found under src/mlp_utils.py

### CIFAR-10 example
The MLP layers have been implemented in an example under MLP_Mixer.ipynb on the CIFAR-10 dataset. Even though the authors of the paper say

There is another Layer implemeted called the Projection layer. Its role is the same as GlobalAvgPooling1D layer is supposed to do according to the original paper, which is "Token Mixing". The only difference is that the projection layer can learn how to weight each channel rather than just the mean.


There is a third run done on the CIFAR-10 dataset, where the inputs to the MLPBlock layer are permuted before the next block. The idea is that this allows the MLPBlocks to do both Channel and Token Mixing.
