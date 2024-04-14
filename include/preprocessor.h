#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <fstream>
#include <execution>
#include <ranges>
#include <unordered_set>
#include <numeric>
#include <torch/torch.h>
#include "logging.h"
#include "classifier.h"

static const int DEPTH_WIDTH = 518;
static const int DEPTH_HEIGHT = 518;

static const int OBJD_WIDTH = 640;
static const int OBJD_HEIGHT = 640;

static const int TRY_COUNT = 2;

struct PreprocessorOptions {
    static const int DEPTH = 1 << 0; 
    static const int OBJECTS = 1 << 1; 
};

class Preprocessor {
    private:
        std::string output_folder;
        Log log;

        Ort::Env depth_env;
        Ort::SessionOptions depth_session_options;
        Ort::Session depth_session{nullptr};

        Ort::Env objd_env;
        Ort::SessionOptions objd_session_options;
        Ort::Session objd_session{nullptr};

        Eigen::MatrixXf xywh_to_xyxy(const Eigen::MatrixXf x);
        float compute_iou(const Eigen::Matrix<float, 1, 4>& box1, const Eigen::Matrix<float, 1, 4>& box2);
        Eigen::MatrixXf nonmax_suppression(const Eigen::MatrixXf boxes, float iou_threshold, int index = 0);
        Eigen::MatrixXf resize_boxes(const Eigen::MatrixXf& boxes_xyxy, int newWidth, int newHeight, int oldWidth, int oldHeight);
        Ort::Value objd_preprocessor(cv::Mat image);
        Eigen::MatrixXf objd_postprocessor(std::vector<Ort::Value>& output_tensors, float min_confidence = 0.75f);
        torch::Tensor detect_objects(cv::Mat image, float min_confidence = 0.75);
        Ort::Value depth_preprocessor(cv::Mat image);
        float* depth_postprocessor(std::vector<Ort::Value>& output_tensors);
        float* estimate_depth(cv::Mat image);

    public:
        Preprocessor(std::string model_folder = "./Models", std::string output_folder = "./", LogCallback callback = nullptr);
        ~Preprocessor();
        torch::Tensor run(cv::Mat image, bool from_unity = false, int options = PreprocessorOptions::DEPTH | PreprocessorOptions::OBJECTS);
};

#endif // PREPROCESSOR_H