// features/ColorCorrelogram.cpp
#include "ColorCorrelogram.h"
using namespace std;

vector<float> ColorCorrelogram::extract(const cv::Mat& image) const {
    // Tiền xử lý ảnh
    cv::Mat processed = preprocess(image, false);
	// Chuyển đổi ảnh sang không gian màu HSV
	cv::Mat quantized = cv::Mat::zeros(processed.size(), CV_8UC1); // CV_8UC1: 8-bit unsigned single-channel image

    // Lượng hóa không gian màu về 64 trạng thái cố định
    for (int r = 0; r < processed.rows; ++r) {
        for (int c = 0; c < processed.cols; ++c) {
            cv::Vec3b p = processed.at<cv::Vec3b>(r, c);
            quantized.at<uchar>(r, c) = (p[0] / 64) + (p[1] / 64) * 4 + (p[2] / 64) * 16;
        }
    }

	vector<float> correlogram(64, 0.0f); // 64 trạng thái màu
	vector<float> total_count(64, 0.0f); // Đếm tổng số pixel cho mỗi trạng thái màu
	int dr[] = { -1, 1, 0, 0 }; // Hàng xung quanh (trên, dưới, trái, phải)
	int dc[] = { 0, 0, -1, 1 }; // Cột xung quanh (trên, dưới, trái, phải)

	// Với mỗi pixel, kiểm tra các pixel xung quanh và tính toán số lượng pixel cùng màu
    for (int r = 1; r < quantized.rows - 1; ++r) {
        for (int c = 1; c < quantized.cols - 1; ++c) {
            uchar color = quantized.at<uchar>(r, c);
            total_count[color] += 4.0f;
            for (int i = 0; i < 4; ++i) {
                if (quantized.at<uchar>(r + dr[i], c + dc[i]) == color) {
                    correlogram[color] += 1.0f;
                }
            }
        }
    }

	// Chuẩn hóa giá trị trong correlogram
	// chia cho tổng số pixel cùng màu để có giá trị trung bình
    for (int i = 0; i < 64; ++i) {
        if (total_count[i] > 0) correlogram[i] /= total_count[i];
    }
    return correlogram;
}

string ColorCorrelogram::name() const { return "ColorCorrelogram"; }