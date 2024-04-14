#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include "residualblock.h"

struct ModelImpl : torch::nn::Module {
    ModelImpl();
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    torch::nn::Conv2d downsample{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};
    ResidualBlock
        block1{nullptr}, 
        block2{nullptr}, 
        block3{nullptr}, 
        block4{nullptr}, 
        block5{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc{nullptr};
    torch::nn::Sigmoid std_dev_head{nullptr};
    torch::nn::Tanh mean_head{nullptr};
    torch::nn::Sequential critic_head{nullptr};
};

TORCH_MODULE(Model);
// From https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/deep_residual_network/include/resnet.h : 
// Wrap class into ModuleHolder (a shared_ptr wrapper),
// see https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/pimpl.h
// class Model : public torch::nn::ModuleHolder<ModelImpl> {
//  public:
//     using torch::nn::ModuleHolder<ModelImpl>::ModuleHolder;
// };

#endif