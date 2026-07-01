// features/SIFT.h
#pragma once
#include "../core/Feature.h"
#include "../indexing/BoW.h"
using namespace std;

/**
 * @brief Lớp trích xuất đặc trưng điểm bất biến cấu trúc cục bộ SIFT kết hợp BoW
 */
class SIFTFeature : public Feature {
private:
    BoWVocabulary bow;
public:
    SIFTFeature();
    vector<float> extract(const cv::Mat& image) const override;
    string name() const override;
};