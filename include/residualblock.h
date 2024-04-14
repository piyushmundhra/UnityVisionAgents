#ifndef RESIDUALBLOCK_H
#define RESIDUALBLOCK_H

#include <torch/torch.h>

struct ResidualBlockImpl : torch::nn::Module {
    ResidualBlockImpl(int16_t in_channels, int16_t out_channels, int16_t stride = 1);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential shortcut{nullptr};
};

TORCH_MODULE(ResidualBlock);

#endif