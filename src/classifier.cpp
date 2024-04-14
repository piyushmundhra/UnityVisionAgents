#include <classifier.h>

ClassifierImpl::ClassifierImpl():
    downsample(register_module("downsample", torch::nn::Conv2d(torch::nn::Conv2dOptions(2 * INPUT_FRAME_PAIRS, 32, 7).stride(2).padding(3)))),
    maxpool(register_module("maxpool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)))),
    block1(register_module("block1", ResidualBlock(64, 64))),
    block2(register_module("block2", ResidualBlock(64, 64))),
    block3(register_module("block3", ResidualBlock(64, 128, 2))),
    block4(register_module("block4", ResidualBlock(128, 128))),
    block5(register_module("block5", ResidualBlock(128, 128))),
    avgpool(register_module("avgpool", torch::nn::AdaptiveAvgPool2d(1))),
    action_head(register_module("action_head", torch::nn::Sequential(
        torch::nn::Linear(128, YAW_ACTUATIONS.size() * MOVEMENT_ACTUATIONS.size()),
        torch::nn::Softmax(1)
    ))),
    q_value_head(register_module("q_value_head", torch::nn::Linear(128, YAW_ACTUATIONS.size() * MOVEMENT_ACTUATIONS.size())))
{}

std::tuple<torch::Tensor, torch::Tensor> ClassifierImpl::forward(torch::Tensor x) {
    x = downsample->forward(x);
    x = maxpool->forward(x);
    x = block1->forward(x);
    x = block2->forward(x);
    x = block3->forward(x);
    x = block4->forward(x);
    x = block5->forward(x);
    x = avgpool->forward(x);
    x = x.view({x.size(0), -1});
    torch::Tensor x1 = action_head->forward(x); 
    torch::Tensor x2 = q_value_head->forward(x);
    return {x1,x2};
}

std::string ClassifierImpl::test_forward_pass() {
    std::stringstream ss;
    torch::Tensor x = torch::randn({1, 2 * INPUT_FRAME_PAIRS, DQN_HEIGHT, DQN_HEIGHT});
    try {
        ss << "Input size: " << x.sizes() << "\n";

        x = downsample->forward(x);
        ss << "After downsample: " << x.sizes() << "\n";

        x = maxpool->forward(x);
        ss << "After maxpool: " << x.sizes() << "\n";

        x = block1->forward(x);
        ss << "After block1: " << x.sizes() << "\n";

        x = block2->forward(x);
        ss << "After block2: " << x.sizes() << "\n";

        x = block3->forward(x);
        ss << "After block3: " << x.sizes() << "\n";

        x = block4->forward(x);
        ss << "After block4: " << x.sizes() << "\n";

        x = block5->forward(x);
        ss << "After block5: " << x.sizes() << "\n";

        x = avgpool->forward(x);
        ss << "After avgpool: " << x.sizes() << "\n";

        x = x.view({x.size(0), -1});
        ss << "After view: " << x.sizes() << "\n";

        torch::Tensor x1 = action_head->forward(x); 
        ss << "After action_head: " << x1.sizes() << "\n";

        torch::Tensor x2 = q_value_head->forward(x);
        ss << "After q_value_head: " << x2.sizes() << "\n";
      } catch (const std::exception& e) {
        ss << "Exception caught during forward pass: " << e.what() << "\n";
    } catch (...) {
        ss << "Unknown exception caught during forward pass.\n";
    }
    return ss.str();
}
