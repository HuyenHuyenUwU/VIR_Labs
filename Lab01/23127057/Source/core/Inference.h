// core/Inference.h
#pragma once
#include "FeatureDatabase.h"
#include "Retrieval.h"
#include "../features/Blob.h"
#include "../features/ColorHistogram.h"
#include "../features/ColorCorrelogram.h"
#include "../features/Corner.h"
#include "../features/SIFT.h"
#include "../features/ORB.h"
#include "../features/Texture.h"
#include <chrono>
using namespace std;

// Cấu trúc đóng gói dữ liệu đầu ra của một lượt tìm kiếm chuyên nghiệp
struct InferenceOutput {
    vector<SearchResult> results;      // Danh sách Top K ảnh kèm điểm số
    double executionTimeMs;            // Thời gian tìm kiếm thực tế (ms)
};

class InferenceEngine {
private:
    BlobFeature m_blobExtractor;
    ColorHistogram m_colorHistogramExtractor;
    ColorCorrelogram m_colorCorrelogramExtractor;
    CornerFeature m_cornerExtractor;
    SIFTFeature m_siftExtractor;
    ORBFeature m_orbExtractor;
    TextureFeature m_textureExtractor;

    FeatureDatabase m_blobDB;
    FeatureDatabase m_colorHistogramDB;
    FeatureDatabase m_colorCorrelogramDB;
    FeatureDatabase m_cornerDB;
    FeatureDatabase m_siftDB;
    FeatureDatabase m_orbDB;
    FeatureDatabase m_textureDB;
    bool m_isInitialized = false;

public:
    /**
     * @brief Nạp toàn bộ file dữ liệu đặc trưng .yml vào bộ nhớ RAM (Chỉ làm 1 lần lúc bật chương trình)
     * @param basePath Đường dẫn cơ sở chứa các file db_*.yml
     */
    bool initialize(const string& basePath);

    /**
     * @brief Thực hiện tìm kiếm thực tế, tính toán thời gian phản hồi sử dụng cả 7 đặc trưng
     * @param queryImage Ảnh truy vấn
     * @param weights Trọng số cho 7 đặc trưng: blob, colorhistogram, colorcorrelogram, corner, sift, orb, texture
     * @param K Số lượng kết quả trả về
     */
    InferenceOutput query(const cv::Mat& queryImage, const vector<float>& weights, int K);
};