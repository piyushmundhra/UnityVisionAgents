#include "agents.h"

ClassificationAgent::ClassificationAgent(std::string model_folder, std::string output_folder, LogCallback callback){
    if (callback != nullptr) {
        this->log = Log(callback);
    }

    this->output_folder = output_folder;

    std::ifstream file(output_folder + "/classifier_weights.pt");
    this->log("Checking for model weights at location: " + output_folder + "/classifier_weights.pt");
    if (file.good()) {
        torch::load(this->classifier, output_folder + "/classifier_weights.pt");
    } else {
        this->log("Model weights file does not exist. Please check the file path. Creating new weights checkpoint\n");
        torch::save(this->classifier, output_folder + "/classifier_weights.pt");
    }

    this->log(this->classifier->test_forward_pass());

    this->model_optimizer = std::make_unique<torch::optim::Adam>(this->classifier->parameters(), torch::optim::AdamOptions(0.005));
    pre = new Preprocessor(model_folder, output_folder, callback);
    this->log("Model, preprocessor, and optimizer loaded.\n");
    this->reset_buffer();
}

ClassificationAgent::~ClassificationAgent(){
    delete pre;
}

void ClassificationAgent::reset_buffer(){
    this->log("Resetting buffer");
    this->labels = torch::empty({0,1}, torch::kLong);
    this->logits = torch::empty({0,YAW_ACTUATIONS.size() * MOVEMENT_ACTUATIONS.size()});
    this->rewards = torch::empty({0,1});
    this->predicted_q_values = torch::empty({0,YAW_ACTUATIONS.size() * MOVEMENT_ACTUATIONS.size()});
}

/**
 * This function infers the action for the given states and stores the result.
 * It first preprocesses the states and stores them in the 'states' member.
 * Then it performs a forward pass through the DQN model and stores the output action in the 'actions' member.
 * The function returns the output action.
 *
 * IMPORTANT: after every call to this function, the 'follow_up' function must be called to add the correct classifications and q values.
 *
 * @param rgb_frames A vector of pointers to the raw frame data.
 * @param depth_frames A vector of pointers to the depth frame data.
 * @param height The height of each frame (assumes square frame).
 * @return The output action as a tensor.
 */
torch::Tensor ClassificationAgent::infer_and_store(unsigned char** rgb_frames, float** depth_frames, int height){
    torch::Tensor input = torch::empty({0, DQN_HEIGHT, DQN_HEIGHT});
    for (int i = 0; i < INPUT_FRAME_PAIRS; i++){

        cv::Mat inputMat(cv::Size(height, height), CV_8UC3, rgb_frames[i]);
        if (depth_frames != nullptr){

            torch::Tensor depth = torch::from_blob(depth_frames[i], {1, 1, height, height});
            depth = depth.flip({2});
            this->log("Created depth tensor");
            std::vector<int64_t> size = {DQN_HEIGHT, DQN_HEIGHT};
            depth = torch::nn::functional::interpolate(
                depth, 
                torch::nn::functional::InterpolateFuncOptions()
                    .size(size)
                    .mode(torch::kBilinear)
                ).squeeze(0);
            this->log("Downsampled depth tensor");
            input = torch::cat({input, depth}, 0);
            input = torch::cat({input, pre->run(inputMat, true, PreprocessorOptions::OBJECTS)}, 0);
        } else {
            input = torch::cat({input, pre->run(inputMat, true, PreprocessorOptions::OBJECTS | PreprocessorOptions::DEPTH)}, 0);
        }
    }

    this->log("Preprocessed input frames");

    for (int i = 0; i < input.size(0); i++){
        cv::Mat mat(DQN_HEIGHT, DQN_HEIGHT, CV_32FC1, input[i].data_ptr());
        cv::imwrite(this->output_folder + "/input_channel_" + std::to_string(i) + ".png", mat * 255);
    }

    auto output = classifier->forward(input.unsqueeze(0));

    torch::Tensor q_values = std::get<1>(output);
    this->predicted_q_values = torch::cat({this->predicted_q_values, q_values}, 0);
    
    torch::Tensor action = std::get<0>(output);
    this->logits = torch::cat({this->logits, action}, 0);

    int predicted_label = action.argmax(1).item().toInt();
    this->log("Predicted label: " + std::to_string(predicted_label));
    int yaw_label = predicted_label / MOVEMENT_ACTUATIONS.size();
    int movement_label = predicted_label % MOVEMENT_ACTUATIONS.size();

    torch::Tensor predicted_actions = torch::tensor({YAW_ACTUATIONS[yaw_label], MOVEMENT_ACTUATIONS[movement_label]}, torch::kLong).view({1,2});
    this->log("Actuation output: " + std::to_string(YAW_ACTUATIONS[yaw_label]) + ", " + std::to_string(MOVEMENT_ACTUATIONS[movement_label]));
    return predicted_actions;
}

void ClassificationAgent::follow_up(float start_angle, float start_distance, float new_angle, float new_distance){
    torch::Tensor reward = torch::tensor({this->reward(new_angle, new_distance)}).view({1,1});
    torch::Tensor label = torch::tensor({this->optimal_action(start_angle, start_distance)}, torch::kLong).view({1,1});
    this->log("follow_up: \tangle1: " + std::to_string(start_angle) + "\tdistance1: " + std::to_string(start_distance));
    this->log("follow_up: \tangle2: " + std::to_string(new_angle) + "\tdistance2: " + std::to_string(new_distance));
    this->log("label:" + std::to_string(label.item<float>()));
    this->log("reward:" + std::to_string(reward.item<float>()));

    this->rewards = torch::cat({this->rewards, reward}); 
    this->labels = torch::cat({this->labels, label});
}

int ClassificationAgent::optimal_action(float angle, float distance){
    float bestDist = std::abs(distance);
    int distIdx = 0;
    float bestAngle = std::abs(angle);
    int angleIdx = 0;
    for (int i = 0; i < YAW_ACTUATIONS.size(); i++){
        if (std::abs(YAW_ACTUATIONS[i] + angle) < bestAngle){
            bestAngle = std::abs(YAW_ACTUATIONS[i] + angle);
            angleIdx = i;
        }
    }
    for (int i = 0; i < MOVEMENT_ACTUATIONS.size(); i++){
        if (std::abs(MOVEMENT_ACTUATIONS[i] + distance) < bestDist){
            bestDist = std::abs(MOVEMENT_ACTUATIONS[i] + distance);
            distIdx = i;
        }
    }
    this->log("angleIdx: " + std::to_string(angleIdx) + ", distIdx: " + std::to_string(distIdx));
    int label = MOVEMENT_ACTUATIONS.size() * angleIdx + distIdx;
    return label;
}

float ClassificationAgent::reward(float angle, float distance){
    int reward = 0;
    if (std::abs(angle) < 15){
        reward += 0.5;
    }
    if (std::abs(distance - OPTIMAL_DISTANCE) < 0.5){
        reward += 0.5;
    }
    return reward;
}

void ClassificationAgent::bootstrap_train() {
    this->log("Starting training");
    this->log("Logits tensor " + tensorToString(this->logits));
    this->log("labels tensor " + tensorToString(this->labels));

    torch::Tensor loss = torch::nn::functional::cross_entropy(this->logits, this->labels.squeeze());
  
    this->model_optimizer->zero_grad();

    double grad_sum = 0.0;
    for (const auto& pair : this->classifier->named_parameters()) {
        const auto& grad = pair.value().grad();
        if (grad.defined()) {
            grad_sum += grad.sum().item<double>();
        }
    }

    torch::Tensor predicted_q_values_actions_taken = this->predicted_q_values.gather(1, this->labels);
    torch::Tensor target_q_values = compute_target_q_values();
    torch::Tensor q_value_loss = torch::nn::functional::mse_loss(predicted_q_values_actions_taken, target_q_values);
    torch::Tensor total_loss = loss + q_value_loss;
    total_loss.backward();

    this->log("Sum of weight gradients: " + std::to_string(grad_sum));
    this->model_optimizer->step();
    this->log("Training loss: " + std::to_string(loss.item<float>()));
    this->log("Q value loss: " + std::to_string(q_value_loss.item<float>()));
    torch::save(this->classifier, output_folder + "/classifier_weights.pt");
    this->reset_buffer();
}

void ClassificationAgent::reinforcement_train(){

}

extern "C" {

    __attribute__((visibility("default"))) ClassificationAgent* create_classifier(const char* model_folder, const char* output_folder, LogCallback callback) {
        return new ClassificationAgent(model_folder, output_folder, callback);
    }

    __attribute__((visibility("default"))) void delete_classifier(ClassificationAgent* classifier){
        delete classifier;
    }

    __attribute__((visibility("default"))) void test_depth_data(ClassificationAgent* classifier, float* depth_data, int width, int height){
        classifier->log("testing depth data");
        cv::Mat mat = cv::Mat(height, width, CV_32FC1, depth_data);
        cv::flip(mat, mat, 0);
        mat = mat * 255;
        cv::imwrite(classifier->output_folder + "/depth_data.png", mat);
    }
 
    __attribute__((visibility("default"))) void infer_and_store_classifier(ClassificationAgent* classifier, unsigned char** rgb_frames, float** depth_frames, int height, int* output_array) {
        torch::Tensor output = classifier->infer_and_store(rgb_frames, depth_frames, height);
        output = output.squeeze();
        output_array[0] = output[0].item<int>();
        output_array[1] = output[1].item<int>();
    }

    __attribute__((visibility("default"))) void follow_up_classifier(ClassificationAgent* classifier, float start_angle, float start_distance, float new_angle, float new_distance){
        classifier->follow_up(start_angle, start_distance, new_angle, new_distance);
    }

    __attribute__((visibility("default"))) void bootstrap_train(ClassificationAgent* classifier){
        classifier->bootstrap_train();
    }

    __attribute__((visibility("default"))) void reinforcement_train(ClassificationAgent* classifier){
        classifier->reinforcement_train();
    }
}