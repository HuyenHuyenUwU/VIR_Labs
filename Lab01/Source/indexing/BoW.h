// indexing/BoW.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

/**
 * @brief Lớp Bag of Words quản lý việc lượng hóa các đặc trưng cục bộ thành vector tần suất cố định
 */
class BoWVocabulary {
public:
    cv::Mat vocabulary;       /**< Ma trận tâm cụm từ vựng thị giác */
    int vocabularySize = 100; /**< Số lượng từ vựng cố định */

    /**
     * @brief Khởi tạo từ vựng giả lập ngẫu nhiên phục vụ chạy thử nghiệm hệ thống
     */
    void createDummyVocabulary(int descriptorDims);

    /**
     * @brief Ánh xạ và lượng hóa danh sách đặc trưng cục bộ thành vector phân bố 1 chiều.
     */
    vector<float> quantize(const cv::Mat& descriptors) const;
};