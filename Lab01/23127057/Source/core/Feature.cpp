// core/Feature.cpp
#include "Feature.h"
#include <cmath>
#include <algorithm>

float Feature::similarity(const std::vector<float>& a, const std::vector<float>& b) const {
	// Nếu một trong hai vector rỗng hoặc có kích thước khác nhau, trả về 0.0f
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;

	// Tính tích vô hướng (dot product) và chuẩn (norm) của hai vector
    double dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA == 0.0 || normB == 0.0) return 0.0f;
	// Khoảng cách cosine similarity = dot(A, B) / (||A|| * ||B||)
    return static_cast<float>(dotProduct / (std::sqrt(normA) * std::sqrt(normB)));
}

cv::Mat Feature::preprocess(const cv::Mat& image, bool toGray) const {
    if (image.empty()) return image;
    cv::Mat processed;
	// Resize về TARGET_SIZE x TARGET_SIZE = 256 x 256
    cv::resize(image, processed, cv::Size(TARGET_SIZE, TARGET_SIZE));

	// Nếu toGray = true và ảnh có 3 kênh (BGR), chuyển sang grayscale và làm mờ Gaussian
    if (toGray && processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(processed, processed, cv::Size(3, 3), 0);
    }
	// Nếu toGray = false và ảnh có 3 kênh (BGR), áp dụng filter bilateral để giảm nhiễu nhưng vẫn giữ cạnh
    else if (!toGray && processed.channels() == 3) {
        cv::Mat filtered;
        cv::bilateralFilter(processed, filtered, 5, 75, 75);
        processed = filtered;
    }
    return processed;
}