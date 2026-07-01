// indexing/BoW.cpp
#include "BoW.h"
#include <limits>
using namespace std;

void BoWVocabulary::createDummyVocabulary(int descriptorDims) {
	// Tạo một ma trận từ vựng giả lập với các giá trị ngẫu nhiên
	vocabulary = cv::Mat::ones(vocabularySize, descriptorDims, CV_32FC1); // CV_32FC1: ma trận 32-bit float với 1 kênh
    cv::randu(vocabulary, cv::Scalar(0), cv::Scalar(255));
}

// tạo vector BoW từ các đặc trưng cục bộ
vector<float> BoWVocabulary::quantize(const cv::Mat& descriptors) const {
    vector<float> bowVector(vocabularySize, 0.0f);
    if (descriptors.empty() || vocabulary.empty()) return bowVector;

    cv::Mat floatDescriptors;
    descriptors.convertTo(floatDescriptors, CV_32FC1);

    for (int i = 0; i < floatDescriptors.rows; ++i) {
		// tìm cái từ vựng gần nhất (min_dist) với đặc trưng cục bộ hiện tại
        float min_dist = numeric_limits<float>::max();
        int best_cluster = 0;
        for (int j = 0; j < vocabulary.rows; ++j) {
            float dist = static_cast<float>(cv::norm(floatDescriptors.row(i), vocabulary.row(j), cv::NORM_L2));
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        bowVector[best_cluster] += 1.0f;
    }

	// Chuẩn hóa vector BoW để tổng các phần tử bằng 1
    float sum = 0.0f;
    for (float val : bowVector) sum += val;
    if (sum > 0.0f) {
        for (float& val : bowVector) val /= sum;
    }
    return bowVector;
}