#include <preprocessor.h>

Preprocessor::Preprocessor(std::string model_folder, std::string output_folder, LogCallback callback) {
    this->output_folder = output_folder;
    if (callback != nullptr) {
        this->log = Log(callback);
    } 

    std::string depth_model_path = model_folder + "/depth.onnx";
    this->depth_session = Ort::Session(depth_env, depth_model_path.c_str(), depth_session_options);

    std::string objd_model_path = model_folder + "/yolov8s.onnx";
    this->objd_session = Ort::Session(objd_env, objd_model_path.c_str(), objd_session_options);
}

Preprocessor::~Preprocessor(){
    depth_session.release();
    objd_session.release();
}

Eigen::MatrixXf Preprocessor::xywh_to_xyxy(const Eigen::MatrixXf x) {
    Eigen::MatrixXf _x = x;

    _x.col(0) = x.col(0) - x.col(2) / 2.0;
    _x.col(1) = x.col(1) - x.col(3) / 2.0;
    _x.col(2) = x.col(0) + x.col(2) / 2.0;
    _x.col(3) = x.col(1) + x.col(3) / 2.0;

    return _x;
}

float Preprocessor::compute_iou(const Eigen::Matrix<float, 1, 4>& box1, const Eigen::Matrix<float, 1, 4>& box2) {

    float x1 = std::max(box1(0), box2(0));
    float y1 = std::max(box1(1), box2(1));
    float x2 = std::min(box1(2), box2(2));
    float y2 = std::min(box1(3), box2(3));

    float box1_area = (box1(2) - box1(0) + 1) * (box1(3) - box1(1) + 1);
    float box2_area = (box2(2) - box2(0) + 1) * (box2(3) - box2(1) + 1);

    float intersection_area = std::max(0.0f, x2 - x1 + 1) * std::max(0.0f, y2 - y1 + 1);
    float union_area = box1_area + box2_area - intersection_area;

    float iou = intersection_area / union_area;

    return iou;
}

Eigen::MatrixXf Preprocessor::nonmax_suppression(const Eigen::MatrixXf boxes, float iou_threshold, int index){
    if (index == boxes.rows()){
        return boxes;
    }

    Eigen::MatrixXf filtered_boxes(boxes.rows(), boxes.cols());
    int boxes_added = index + 1;

    for (int i = 0; i < boxes_added; i++){
        filtered_boxes.row(i) = boxes.row(i);
    }

    for (int j = boxes_added; j < boxes.rows(); ++j) {
        float iou = compute_iou(boxes.row(index), boxes.row(j));
        if (iou < iou_threshold) {
            filtered_boxes.row(boxes_added) = boxes.row(j);
            boxes_added++;
        }
    }

    filtered_boxes.conservativeResize(boxes_added, Eigen::NoChange);
    return filtered_boxes;
}

Eigen::MatrixXf Preprocessor::resize_boxes(const Eigen::MatrixXf& boxes_xyxy, int newWidth, int newHeight, int oldWidth, int oldHeight) {
    Eigen::MatrixXf denom(boxes_xyxy.rows(), 4);
    denom.col(0) = Eigen::VectorXf::Constant(boxes_xyxy.rows(), oldWidth);
    denom.col(1) = Eigen::VectorXf::Constant(boxes_xyxy.rows(), oldHeight);
    denom.col(2) = Eigen::VectorXf::Constant(boxes_xyxy.rows(), oldWidth);
    denom.col(3) = Eigen::VectorXf::Constant(boxes_xyxy.rows(), oldHeight);

    Eigen::MatrixXf numer(boxes_xyxy.rows(), 4);
    numer.col(0) = Eigen::VectorXf::Constant(boxes_xyxy.rows(), newWidth);
    numer.col(1) = Eigen::VectorXf::Constant(boxes_xyxy.rows(), newHeight);
    numer.col(2) = Eigen::VectorXf::Constant(boxes_xyxy.rows(), newWidth);
    numer.col(3) = Eigen::VectorXf::Constant(boxes_xyxy.rows(), newHeight);

    Eigen::MatrixXf resized_boxes = boxes_xyxy.array() / denom.array() * numer.array();

    return resized_boxes;
}

Ort::Value Preprocessor::objd_preprocessor(cv::Mat image){
    cv::resize(image, image, cv::Size(OBJD_WIDTH, OBJD_HEIGHT));
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, OBJD_HEIGHT, OBJD_WIDTH};
    return Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());
}

Eigen::MatrixXf Preprocessor::objd_postprocessor(std::vector<Ort::Value>& output_tensors, float min_confidence){
    auto output_tensor = output_tensors.front().GetTensorMutableData<float>();

    int cols = 84;
    int rows = 8400;

    Eigen::MatrixXf mat = Eigen::Map<Eigen::MatrixXf>(output_tensor, rows, cols);
    Eigen::MatrixXf class_info = mat.block(0, 4, mat.rows(), mat.cols() - 4);
    Eigen::MatrixXf scores = class_info.rowwise().maxCoeff();
    Eigen::MatrixXf boxes_xywh = mat.block(0, 0, mat.rows(), 4);

    Eigen::Array<bool, Eigen::Dynamic, 1> mask = scores.array() > min_confidence;
    int count = mask.count();
    this->log("valid boxes after min_confidence masking: " + std::to_string(count) + " out of " + std::to_string(mask.rows()));
    std::unordered_set<int> hashtable;
    for (int i = mask.rows() - 1; i >= 0; i--){
        bool b = mask(i);
        if (b){
            hashtable.insert(i);
        }
        if (hashtable.size() == count){
            break;
        }
    }    

    Eigen::MatrixXf filtered_boxes_xywh(count, 4);
    Eigen::MatrixXf filtered_scores(count, 1);

    int row_index = 0;
    for (auto it = hashtable.begin(); it != hashtable.end(); ++it) {
        int classification;
        // https://stackoverflow.com/questions/62280277/argmax-method-in-c-eigen-library
        class_info.row(*it).maxCoeff(&classification);
        // class 0 = person
        if (classification == 0) {
            filtered_boxes_xywh.row(row_index) = boxes_xywh.row(*it);
            filtered_scores.row(row_index) = scores.row(*it);
            row_index++;
        }
    }
    
    Eigen::MatrixXf boxes = xywh_to_xyxy(filtered_boxes_xywh);

    // Sort boxes based on filtered_scores
    std::vector<int> indices(filtered_scores.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(
        indices.begin(), 
        indices.end(), 
        [&](int i, int j) {
            return filtered_scores(i, 0) > filtered_scores(j, 0);
        }
    );

    Eigen::MatrixXf sorted_boxes(boxes.rows(), boxes.cols());
    for (int i = 0; i < indices.size(); ++i) {
        sorted_boxes.row(i) = boxes.row(indices[i]);
    }

    Eigen::MatrixXf final_boxes = nonmax_suppression(sorted_boxes, 0.5f);
    // this->log("valid boxes after nonmax suppresion: " + std::to_string(final_boxes.rows()));
    return final_boxes;
}

torch::Tensor Preprocessor::detect_objects(cv::Mat image, float min_confidence){
    Eigen::MatrixXf boxes;
    torch::Tensor mask_tensor = torch::zeros({DQN_HEIGHT, DQN_HEIGHT});

    for (int i = 0; i < TRY_COUNT; ++i) {
        auto input_tensor = objd_preprocessor(image);

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};
        std::vector<Ort::Value> output_tensors = this->objd_session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        boxes = objd_postprocessor(output_tensors);

        if (boxes.rows() > 0) {
            break;
        }
    }

    Eigen::MatrixXf resized_boxes = resize_boxes(boxes, DQN_HEIGHT, DQN_HEIGHT, OBJD_WIDTH, OBJD_HEIGHT);

    for (int i = 0; i < resized_boxes.rows(); i++) {
        float x1 = resized_boxes(i, 0);
        float y1 = resized_boxes(i, 1);
        float x2 = resized_boxes(i, 2);
        float y2 = resized_boxes(i, 3);

        int ix1 = static_cast<int>(std::round(x1));
        int iy1 = static_cast<int>(std::round(y1));
        int ix2 = static_cast<int>(std::round(x2));
        int iy2 = static_cast<int>(std::round(y2));

        ix1 = std::max(0, std::min(ix1, DQN_HEIGHT - 1));
        iy1 = std::max(0, std::min(iy1, DQN_HEIGHT - 1));
        ix2 = std::max(0, std::min(ix2, DQN_HEIGHT - 1));
        iy2 = std::max(0, std::min(iy2, DQN_HEIGHT - 1));

        mask_tensor.slice(0, iy1, iy2 + 1).slice(1, ix1, ix2 + 1).fill_(1);
    }

    return mask_tensor;
}  

Ort::Value Preprocessor::depth_preprocessor(cv::Mat image){
    auto start = std::chrono::high_resolution_clock::now();
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(DQN_HEIGHT, DEPTH_HEIGHT), 0, 0, cv::INTER_CUBIC);

    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(resized_image, mean, resized_image);
    cv::divide(resized_image, std, resized_image);

    cv::Mat blob;
    cv::dnn::blobFromImage(resized_image, blob);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, 518, 518};
    return Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());
}

float* Preprocessor::depth_postprocessor(std::vector<Ort::Value>& output_tensors){
    auto output_tensor = output_tensors.front().GetTensorMutableData<float>();

    float min = *std::min_element(output_tensor, output_tensor + DQN_HEIGHT * DEPTH_HEIGHT);
    float max = *std::max_element(output_tensor, output_tensor + DQN_HEIGHT * DEPTH_HEIGHT);
    for (int i = 0; i < DEPTH_WIDTH * DEPTH_HEIGHT; i++) {
        output_tensor[i] = (output_tensor[i] - min) / (max - min);
    }

    return output_tensor;
}

float* Preprocessor::estimate_depth(cv::Mat image){
    auto input_tensor = depth_preprocessor(image);

    const char* input_names[] = {"image"};
    const char* output_names[] = {"depth"};
    auto output_tensors = this->depth_session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    // std::cout << "Inference completed" << std::endl;

    return depth_postprocessor(output_tensors);
}

torch::Tensor Preprocessor::run(cv::Mat image, bool from_unity, int options){
    if (from_unity) {
        cv::flip(image, image, 0);
    }
    cv::imwrite(this->output_folder + "/rgb_input.png", image);
    torch::Tensor result = torch::empty({0, DQN_HEIGHT, DQN_HEIGHT}); 
    
    if (options & PreprocessorOptions::DEPTH){
        float* depth = estimate_depth(image);
        cv::Mat depth_image (DEPTH_HEIGHT, DQN_HEIGHT, CV_32FC1, depth);
        cv::resize(depth_image, depth_image, cv::Size(DQN_HEIGHT, DQN_HEIGHT), cv::INTER_CUBIC);
        torch::Tensor depth_tensor = torch::from_blob(depth_image.data, {DQN_HEIGHT, DQN_HEIGHT}, torch::kFloat32);
        result = torch::cat({result, depth_tensor.unsqueeze(0)}, 0);
    }

    if (options & PreprocessorOptions::OBJECTS){
        torch::Tensor boxes = detect_objects(image, 0.75f);
        result = torch::cat({result, boxes.unsqueeze(0)}, 0);
    }

    return result;
}