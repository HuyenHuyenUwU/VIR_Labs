// features/Blob.h
#pragma once
#include "../core/Feature.h"
using namespace std;

/**
 * @brief Lớp trích xuất phân bố cấu trúc hình dáng vùng đốm (Blob Feature)
 */
class BlobFeature : public Feature {
public:
    vector<float> extract(const cv::Mat& image) const override;
    string name() const override;
};