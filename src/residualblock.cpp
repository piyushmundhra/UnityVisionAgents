#include <residualblock.h>

ResidualBlockImpl::ResidualBlockImpl(int16_t in_channels, int16_t out_channels, int16_t stride): 
    conv1(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false)))),
    bn1(register_module("bn1", torch::nn::BatchNorm2d(out_channels))),
    conv2(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1).bias(false)))),
    bn2(register_module("bn2", torch::nn::BatchNorm2d(out_channels)))
{
    if (stride != 1 || in_channels != out_channels) {
        shortcut = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
            torch::nn::BatchNorm2d(out_channels)
        );
    } else {
        shortcut = torch::nn::Sequential();
    }
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    torch::Tensor out = conv1->forward(x);
    out = bn1->forward(out);
    out = torch::relu(out);
    out = conv2->forward(out);
    out = bn2->forward(out);
    if (!shortcut->is_empty()) {
        out += shortcut->forward(x);
    }
    out = torch::relu(out);
    return out;
}