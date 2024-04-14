#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <torch/torch.h>
#include "residualblock.h"

constexpr std::array<int, 3> YAW_ACTUATIONS = {0, -10, 10}; // degrees
constexpr std::array<int, 3> MOVEMENT_ACTUATIONS = {0, -15, 15}; // cm

const int INPUT_FRAME_PAIRS = 1;
static const int DQN_HEIGHT = 128;


// labels will be interpreted as: [y_0, m_0], ... [y_0, m_n] ... [y_n, m_n]

struct ClassifierImpl : torch::nn::Module {
    ClassifierImpl();
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    std::string test_forward_pass();

    torch::nn::Conv2d downsample{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};
    ResidualBlock
        block1{nullptr}, 
        block2{nullptr}, 
        block3{nullptr}, 
        block4{nullptr}, 
        block5{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Sequential action_head{nullptr};
    torch::nn::Linear q_value_head{nullptr};
};

TORCH_MODULE(Classifier);

#endif