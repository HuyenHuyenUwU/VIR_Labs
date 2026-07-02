// features/SIFT.cpp
#include "SIFT.h"
using namespace std;

SIFTFeature::SIFTFeature() {
    bow.createDummyVocabulary(128); // SIFT mặc định 128 chiều
}

vector<float> SIFTFeature::extract(const cv::Mat& image) const {
    // chuyển sang grayscale
    cv::Mat gray = preprocess(image, true);
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(200);
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    return bow.quantize(descriptors);
}

std::string SIFTFeature::name() const { return "SIFT"; }