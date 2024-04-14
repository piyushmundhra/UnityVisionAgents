#include <agents.h>

ActorCriticAgent::ActorCriticAgent(std::string model_folder, std::string output_folder, LogCallback callback){
    if (callback != nullptr) {
        this->log = Log(callback);
    }

    this->output_folder = output_folder;

    std::ifstream file(output_folder + "/model_weights.pt");
    this->log("Checking for model weights at location: " + output_folder + "/model_weights.pt");
    if (file.good()) {
        torch::load(this->model, output_folder + "/model_weights.pt");
    } else {
        this->log("Model weights file does not exist. Please check the file path. Creating new weights checkpoint\n");
        torch::save(this->model, output_folder + "/model_weights.pt");
    }

    this->model_optimizer = std::make_unique<torch::optim::Adam>(this->model->parameters(), torch::optim::AdamOptions(0.005));
    pre = new Preprocessor(model_folder, output_folder, callback);
    this->log("Model, preprocessor, and optimizer loaded.\n");
    this->reset_buffer();
}

ActorCriticAgent::~ActorCriticAgent(){
    delete pre;
}

void ActorCriticAgent::reset_buffer(){
    this->log("Resetting buffer");
    this->predicted_distributions = torch::empty({0,4});
    this->states = torch::empty({0, 4, 64, 64});
    this->critic_values = torch::empty({0, 1});
    this->rewards = torch::empty({0, 1});
    this->actions = torch::empty({0, 2});
}

/**
 * This function infers the action for the given states and stores the result.
 * It first preprocesses the states and stores them in the 'states' member.
 * Then it performs a forward pass through the DQN model and stores the output action in the 'actions' member.
 * The function returns the output action.
 *
 * IMPORTANT: after every call to this function, the 'follow_up' function must be called to add the corresponding reward.
 *
 * @param frames A vector of pointers to the raw frame data.
 * @param width The width of each frame.
 * @param height The height of each frame.
 * @return The output action as a tensor.
 */
torch::Tensor ActorCriticAgent::infer_and_store(unsigned char** frames, int width, int height){

    std::vector<torch::Tensor> preprocessed_inputs(INPUT_FRAMES);
    #pragma omp parallel for
    for(int i = 0; i < INPUT_FRAMES; i++){
        cv::Mat inputMat(cv::Size(width, height), CV_8UC3, frames[i]);
        torch::Tensor temp = pre->run(inputMat, true);
        preprocessed_inputs[i] = temp;

        // cv::Mat dimg_mat(DQN_HEIGHT, DQN_WIDTH, CV_32FC1, temp[0].data_ptr());
        // cv::imwrite(this->output_folder + "/depth" + std::to_string(i) + ".png", dimg_mat * 255);

        // cv::Mat mask_mat(DQN_HEIGHT, DQN_WIDTH, CV_32FC1, temp[1].data_ptr());
        // cv::imwrite(this->output_folder + "/mask" + std::to_string(i) + ".png", mask_mat * 255);
    }

    torch::Tensor inputTensor = torch::cat(preprocessed_inputs, 0).unsqueeze(0);
    this->states = torch::cat({this->states, inputTensor}, 0);
    
    std::tuple<torch::Tensor, torch::Tensor> temp = this->model->forward(inputTensor);
    torch::Tensor distribution = std::get<0>(temp);
    torch::Tensor value = std::get<1>(temp);

    this->predicted_distributions = torch::cat({this->predicted_distributions, distribution}, 0);
    this->critic_values = torch::cat({this->critic_values, value}, 0);

    torch::Tensor mean_yaw = distribution.select(1, 0);
    torch::Tensor mean_forward = distribution.select(1, 1);
    torch::Tensor std_yaw = distribution.select(1, 2);
    torch::Tensor std_forward = distribution.select(1, 3);

    torch::Tensor action_yaw = mean_yaw + std_yaw * torch::randn({1}, torch::dtype(torch::kFloat32));
    torch::Tensor action_forward = mean_forward + std_forward * torch::randn({1}, torch::dtype(torch::kFloat32));

    torch::Tensor action = torch::cat({action_yaw, action_forward}, 0).view({1,2});
    this->actions = torch::cat({this->actions, action}, 0);
    this->log("Actuation output: \n\t" + tensorToString(action));
    return action;
}

void ActorCriticAgent::follow_up(float reward){
    this->log("Adding reward of " + std::to_string(reward));
    torch::Tensor rewardTensor = torch::tensor({reward}).view({1,1});
    this->rewards = torch::cat({this->rewards, rewardTensor}, 0);
}

// https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
void ActorCriticAgent::train(){
    if (
        states.size(0) != critic_values.size(0) || 
        states.size(0) != predicted_distributions.size(0) || 
        states.size(0) != actions.size(0) || 
        states.size(0) != rewards.size(0)
    ) {
        this->log("Mismatch in 0th dimension size among states, critic_values, predicted_distributions, actions, and rewards");
        this->reset_buffer();
    }

    torch::Tensor mean_yaw = this->predicted_distributions.select(1, 0);
    torch::Tensor mean_forward = this->predicted_distributions.select(1, 1);
    torch::Tensor std_yaw = this->predicted_distributions.select(1, 2);
    torch::Tensor std_forward = this->predicted_distributions.select(1, 3);

    torch::Tensor actions_yaw = this->actions.select(1, 0);
    torch::Tensor actions_forward = this->actions.select(1, 1);

    this->log("mean_yaw: " + tensorToString(mean_yaw));
    this->log("std_yaw: " + tensorToString(std_yaw));

    this->log("mean_forward: " + tensorToString(mean_forward));
    this->log("std_forward: " + tensorToString(std_forward));

    torch::Tensor log_prob_yaw = log_prob(actions_yaw, mean_yaw, std_yaw);
    torch::Tensor log_prob_forward = log_prob(actions_forward, mean_forward, std_forward);
    this->log("log_prob_yaw: " + tensorToString(log_prob_yaw));
    this->log("log_prob_forward: " + tensorToString(log_prob_forward));

    torch::Tensor advantages = this->rewards - this->critic_values.detach();
    this->log("advantages: " + tensorToString(advantages));
    torch::Tensor actor_loss = -((log_prob_yaw + log_prob_forward) * advantages).mean();
    torch::Tensor critic_loss = torch::nn::functional::mse_loss(this->critic_values, this->rewards);
    this->log("critic_loss: " + tensorToString(critic_loss));
    this->log("actor_loss: " + tensorToString(actor_loss));

    torch::Tensor loss = actor_loss + critic_loss;

    this->log("Loss: " + tensorToString(loss) + ", gradients: " + std::to_string(loss.requires_grad()));
    this->log("Computing gradients");

    double weight_sum_before = 0.0;
    for (const auto& p : this->model->parameters()) {
        weight_sum_before += p.abs().sum().item<double>();
    }
    this->log("Sum of weights before optimizer step: " + std::to_string(weight_sum_before));

    this->model_optimizer->zero_grad(); 
    loss.backward();
    this->model_optimizer->step();

    double weight_sum_after = 0.0;
    for (const auto& p : this->model->parameters()) {
        weight_sum_after += p.abs().sum().item<double>();
    }
    this->log("Sum of weights after optimizer step: " + std::to_string(weight_sum_after));

    torch::save(this->model, this->output_folder + "/model_weights.pt");
    this->reset_buffer();
}

const char* ActorCriticAgent::print_model_helper(const torch::nn::Module& module, size_t indentation) {
    std::stringstream ss;

    for (const auto& sub_module : module.named_children()) {
        for (size_t i = 0; i < indentation; ++i) {
            ss << "  ";
        }
        std::stringstream sub_module_ss;
        sub_module.value()->pretty_print(sub_module_ss);
        ss << sub_module_ss.str() << std::endl;
        ss << print_model_helper(*sub_module.value(), indentation + 1);
    }
    
    std::string str = ss.str();
    char* cstr = new char[str.length() + 1];
    std::strcpy(cstr, str.c_str());
    return cstr;
}

const char* ActorCriticAgent::print_model(size_t indentation){
    return print_model_helper(*((this)->model), indentation);
}