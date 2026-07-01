// features/ColorCorrelogram.h
#pragma once
#include "../core/Feature.h"
using namespace std;

/**
 * @brief Lớp trích xuất tương quan không gian màu sắc (Color Correlogram)
 */
class ColorCorrelogram : public Feature {
public:
    vector<float> extract(const cv::Mat& image) const override;
    string name() const override;
};