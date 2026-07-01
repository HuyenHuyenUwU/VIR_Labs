// core/Retrieval.h
#pragma once
#include "Feature.h"
#include "FeatureDatabase.h"
#include "../indexing/InvertedIndex.h"
#include <vector>
#include <string>
using namespace std;

/**
 * @brief Cấu trúc lưu trữ kết quả tìm kiếm của một tấm ảnh.
 */
struct SearchResult {
    string imagePath;      /**< Đường dẫn vật lý của file ảnh */
    float score;           /**< Điểm số độ tương đồng so khớp */
};

/**
 * @brief Lớp điều phối core thực hiện giải thuật tìm kiếm và sắp xếp kết quả
 */
class Retrieval {
public:
    /**
     * @brief Thực hiện tìm kiếm K tấm ảnh giống nhất với ảnh truy vấn
     * @param queryImage Ma trận ảnh truy vấn đầu vào
     * @param extractor Bộ rút trích đặc trưng được lựa chọn
     * @param db Cơ sở dữ liệu đặc trưng toàn cục
     * @param invIndex Inverted index phục vụ tăng tốc tìm kiếm
     * @param K Số lượng kết quả cần trả về
     * @param useInvertedIndex Cờ bật/tắt chế độ tìm kiếm nhanh 
     * @return Danh sách cấu trúc kết quả SearchResult đã được sắp xếp giảm dần
     */
    static vector<SearchResult> search(const cv::Mat& queryImage,
        const Feature& extractor,
        const FeatureDatabase& db,
        const InvertedIndex& invIndex,
        int K,
        bool useInvertedIndex);

    /**
     * @brief Tìm kiếm kết hợp NHIỀU đặc trưng bằng phương pháp Late Fusion (Gộp điểm số)
     * @param queryImage Ảnh truy vấn đầu vào
     * @param extractors Danh sách các con trỏ bộ trích xuất đặc trưng (vd: Màu, Vân bề mặt)
     * @param databases Danh sách các bộ csdl với từng đặc trưng.
     * @param weights Mảng chứa trọng số tỉ lệ của từng đặc trưng (Tổng các trọng số = 1.0)
     * @param K Số lượng ảnh kết quả cần trả về
     */
    static vector<SearchResult> searchLateFusion(const cv::Mat& queryImage,
        const vector<const Feature*>& extractors,
        const vector<const FeatureDatabase*>& databases,
        const vector<float>& weights,
        int K);
};