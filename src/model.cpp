#include <model.h>

ModelImpl::ModelImpl():
    downsample(register_module("downsample", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 32, 3).stride(2).padding(1)))),
    maxpool(register_module("maxpool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)))),
    block1(register_module("block1", ResidualBlock(32, 32))),
    block2(register_module("block2", ResidualBlock(32, 32))),
    block3(register_module("block3", ResidualBlock(32, 64, 2))),
    block4(register_module("block4", ResidualBlock(64, 64))),
    block5(register_module("block5", ResidualBlock(64, 64))),
    avgpool(register_module("avgpool", torch::nn::AdaptiveAvgPool2d(1))),
    fc(register_module("fc", torch::nn::Linear(64, 4))),
    critic_head(register_module("critic_head", torch::nn::Sequential(
        torch::nn::Linear(64, 1),
        torch::nn::Tanh()
    ))),
    std_dev_head(register_module("std_dev_head", torch::nn::Sigmoid())),
    mean_head(register_module("mean_head", torch::nn::Tanh()))
{}

std::tuple<torch::Tensor, torch::Tensor> ModelImpl::forward(torch::Tensor x) {
    x = downsample->forward(x);
    x = maxpool->forward(x);
    x = block1->forward(x);
    x = block2->forward(x);
    x = block3->forward(x);
    x = block4->forward(x);
    x = block5->forward(x);
    x = avgpool->forward(x);
    x = x.view({x.size(0), -1}); // flatten
    torch::Tensor action = fc->forward(x);

    torch::Tensor action1 = action.slice(/*dim=*/1, /*start=*/0, /*end=*/2);
    torch::Tensor action2 = action.slice(/*dim=*/1, /*start=*/2, /*end=*/4);

    torch::Tensor means = mean_head->forward(action1);
    torch::Tensor std_devs = std_dev_head->forward(action1);

    action = torch::cat({means, std_devs}, /*dim=*/1);

    torch::Tensor value = critic_head->forward(x);
    return {action, value};
}

// Input: [1, 4, 64, 64]
// After downsample: [1, 32, 32, 32]
// After maxpool: [1, 32, 16, 16]
// After block1: [1, 32, 16, 16]
// After block2: [1, 32, 16, 16]
// After block3: [1, 64, 8, 8]
// After block4: [1, 64, 8, 8]
// After block5: [1, 64, 8, 8]
// After avgpool: [1, 64, 1, 1]
// After flatten: [1, 64]
// After fc: [1, 4]
// After mean_head: [1, 2]
// After std_dev_head: [1, 2]
// After critic_head: [1, 1]