#pragma once

#include <torch/torch.h>

std::string tensorShapeToString(const torch::Tensor& tensor);
torch::Tensor log_prob(torch::Tensor action, torch::Tensor mean, torch::Tensor std);
std::string tensorToString(const torch::Tensor& tensor);