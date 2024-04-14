#ifndef AGENTS_H
#define AGENTS_H

#include <torch/torch.h>
#include "preprocessor.h"
#include "model.h"
#include "classifier.h"
#include "utils.h"
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <omp.h>

// relevant dqn params
static const int INPUT_FRAMES = 2;
static const int CHANNELS_PER_FRAME = 2;
static const std::tuple<int, int> OUTPUT_SIZE = {1,2};

// training hyperparameters
static const float GAMMA = 0.99;
static const float EPSILON = 0.1;
static const int batch_size = 32;

// reward hyperparameters
static const float ALPHA = 0.66;
static const float BETA = 0.33;
static const float OPTIMAL_DISTANCE = 2; // meters

class ClassificationAgent {
    private:
        Classifier classifier;
        Preprocessor* pre = nullptr;
        std::unique_ptr<torch::optim::Adam> model_optimizer;
        torch::Tensor labels;
        torch::Tensor logits;
        torch::Tensor predicted_q_values;
        torch::Tensor rewards;

        void reset_buffer();
        int optimal_action(float angle, float distance);
        float reward(float angle, float distance);

    public:
        std::string output_folder;
        Log log;

        ClassificationAgent(std::string model_folder, std::string output_folder, LogCallback callback = nullptr);
        ~ClassificationAgent();
        torch::Tensor infer_and_store(unsigned char** frame, float** depth_frames = nullptr, int height = DQN_HEIGHT);
        void follow_up(float start_angle, float start_distance, float new_angle, float new_distance);
        void bootstrap_train();
        void reinforcement_train();
};

class ActorCriticAgent {
    private:    
        Model model;
        Preprocessor* pre = nullptr;
        std::unique_ptr<torch::optim::Adam> model_optimizer;
        std::string output_folder;

        torch::Tensor states;
        torch::Tensor critic_values;
        torch::Tensor predicted_distributions;
        torch::Tensor actions;
        torch::Tensor rewards;

        size_t max_buffer_size = 1000;
        Log log;

        const char* print_model_helper(const torch::nn::Module& module, size_t indentation = 0);
        void reset_buffer();
        
    public:
        ActorCriticAgent(std::string model_folder, std::string output_folder, LogCallback callback = nullptr);
        ~ActorCriticAgent();
        torch::Tensor infer_and_store(unsigned char** frames, int width, int height);
        void follow_up(float reward);
        void train();
        const char* print_model(size_t indentation = 0);
};

#endif // AGENT_H