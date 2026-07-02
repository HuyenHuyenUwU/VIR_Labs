// features/Corner.h
#pragma once
#include "../core/Feature.h"
using namespace std;

/**
 * @brief Lớp trích xuất đặc trưng phân bố mật độ góc cạnh không gian (Corner Feature)
 * Corner được xác định bởi sự thay đổi cường độ sáng theo hai hướng khác nhau 
 */
class CornerFeature : public Feature {
public:
    vector<float> extract(const cv::Mat& image) const override;
    string name() const override;
};