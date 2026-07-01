// evaluation/MAP.h
#pragma once
#include "../core/Retrieval.h"
#include "../core/FeatureDatabase.h"
#include <string>
#include <vector>
using namespace std;

/**
 * @brief Lớp tính toán kiểm thử độ chính xác định dạng giải thuật MAP@K yêu cầu của đề bài
 */
/* @details Giải thuật MAP@K
* 1. Tách nhãn phân lớp (Class Label) từ đường dẫn ảnh truy vấn và các ảnh kết quả
* 2. Tính toán độ chính xác trung bình AP (Average Precision) cho từng ảnh truy vấn
* 3. Tính toán MAP@K bằng cách lấy trung bình các giá trị AP
* 4. Trả về giá trị MAP@K
*/
class MAPEvaluator {
public:
    /**
     * @brief Tách chuỗi trích xuất nhãn phân lớp (Class Label) dựa theo cấu trúc tên đường dẫn
     */
    static string getLabel(const string& path);

    /**
     * @brief Tính toán chỉ số độ chính xác trung bình AP (Average Precision) cho một ảnh mục tiêu tại ngưỡng K[cite: 20, 68].
     */
    static double computeAP(const string& queryPath, const vector<SearchResult>& results, const FeatureDatabase& db, int K);
};