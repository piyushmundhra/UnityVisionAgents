#include "utils.h"

std::string tensorShapeToString(const torch::Tensor& tensor){
    std::ostringstream oss;
    oss << tensor.sizes();
    return oss.str();
}

std::string tensorToString(const torch::Tensor& tensor){
    std::ostringstream oss;
    oss << tensor;
    std::string str = oss.str();
    // std::replace(str.begin(), str.end(), '\n', ' ');
    return str + "\n";
}

torch::Tensor log_prob(torch::Tensor sample, torch::Tensor mean, torch::Tensor std_dev) {
    torch::Tensor variance = std_dev.pow(2);
    torch::Tensor log_scale = (std_dev * torch::sqrt(torch::tensor(2.0 * M_PI))).log();
    return -1 * ((sample - mean).pow(2) / (2 * variance)) - log_scale;
}