// features/ORB.h
#pragma once
#include "../core/Feature.h"
#include "../indexing/BoW.h"
using namespace std;

/**
 * @brief Lớp trích xuất điểm đặc trưng cục bộ tốc độ cao ORB
 * ORB đc xác định bởi sự kết hợp của các đặc trưng FAST keypoint và BRIEF descriptor
 */
class ORBFeature : public Feature {
private:
    BoWVocabulary bow;
public:
    ORBFeature();
    vector<float> extract(const cv::Mat& image) const override;
    string name() const override;
};