#include <iostream>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <opencv2/opencv.hpp>
#include <torch/csrc/jit/mobile/import.h>


//imagenet means for normalization
std::vector<double> norm_mean = {0.485, 0.456, 0.406};
std::vector<double> norm_std = {0.229, 0.224, 0.225};

//Functions
torch::Tensor read_image(const std::string& imageName);
cv::Mat crop_center(const cv::Mat &img);


// Main function
int main(int argc, char* argv[])
{
    // Load the PyTorch model in lite interpreter format.
    torch::jit::mobile::Module model = torch::jit::_load_for_mobile("../weights/binaryclass_optimized_lite_deit.ptl");
    // SImillarly, we can do for multi-class weights


    // input vector to store the image
    std::vector<torch::jit::IValue> inputs;
    at::Tensor in = read_image(argv[1]);
    inputs.push_back(in);

    model.eval();
    // Run the model and capture results in 'class_index'
    at::Tensor class_index = model(inputs).argmax();

    // Print the class index
    // 0 refers to non-drunk and 1 refers to drunk
    if class_index.item() == 0:
        std::cout << "Subject has not consumed the alcohol."<< std::endl;
    else:
        std::cout << "Subject has consumed the alcohol and is not fit for duty."<< std::endl;

    cv::waitKey(0);

    return 0;
}


// Function to read the image and perform certain pre-processing
torch::Tensor read_image(const std::string& imageName)
{
    cv::Mat img = cv::imread(imageName);
    cv::resize(img, img, cv::Size(256,256));
    img = crop_center(img);

    cv::imshow("image", img);

    if (img.channels()==1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    img.convertTo( img, CV_32FC3, 1/255.0 );

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, c10::kFloat);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor.unsqueeze_(0);

    img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);

    return img_tensor.clone();
}


// Function to crop the image
cv::Mat crop_center(const cv::Mat &img)
{
    const int rows = img.rows;
    const int cols = img.cols;

    const int cropSize = std::min(224,224);
    const int offsetW = (cols - cropSize) / 2;
    const int offsetH = (rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);

    return img(roi);
}