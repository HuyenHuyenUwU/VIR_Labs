// core/Feature.h
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
using namespace std;

/**
 * @brief Lớp trừu tượng cơ sở (Abstract base class) định nghĩa giao diện chung cho các thuật toán rút trích đặc trưng
 */
class Feature {
public:
    virtual ~Feature() = default;

    /**
     * @brief Rút trích đặc trưng hình ảnh thành vector 1 chiều.
     * @param image Ảnh gốc đầu vào.
     * @return Vector chứa các giá trị đặc trưng kiểu float.
     */
    virtual vector<float> extract(const cv::Mat& image) const = 0;

    /**
     * @brief Trả về tên định danh duy nhất của thuật toán đặc trưng.
     */
    virtual string name() const = 0;

    /**
     * @brief Tính toán độ tương đồng giữa hai vector đặc trưng bằng khoảng cách Cosine[cite: 66].
     * @param a Vector đặc trưng thứ nhất.
     * @param b Vector đặc trưng thứ hai.
     * @return Giá trị tương đồng thuộc đoạn [-1.0, 1.0].
     */
    virtual float similarity(const vector<float>& a, const vector<float>& b) const;

protected:
    static const int TARGET_SIZE = 256; /**< Kích thước chuẩn hóa hình ảnh */

    /**
     * @brief Tiền xử lý hình ảnh (Chuẩn hóa kích thước, lọc nhiễu, chuyển không gian màu)[cite: 68].
     * @param image Ảnh màu BGR gốc.
     * @param toGray Cờ chỉ định có chuyển đổi sang ảnh xám hay không.
     * @return Ma trận ảnh cv::Mat sau khi đã xử lý lọc nhiễu.
     */
    cv::Mat preprocess(const cv::Mat& image, bool toGray = false) const;
};