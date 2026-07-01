// features/ColorHistogram.h
#pragma once
#include "../core/Feature.h"
using namespace std;

/**
 * @brief Lớp rút trích đặc trưng biểu đồ phân bố màu sắc (Color Histogram) trong không gian màu HSV
 */
class ColorHistogram : public Feature {
public:
    /**
     * @brief Rút trích đặc trưng màu sắc tổng thể từ ảnh đầu vào.
     * @param image Ảnh BGR gốc (đã load bằng imread).
     * @return Vector đặc trưng kiểu float kích thước 256 chiều.
     */
    vector<float> extract(const cv::Mat& image) const override;

    /**
     * @brief Trả về tên định danh duy nhất của đặc trưng.
     */
    string name() const override;
};