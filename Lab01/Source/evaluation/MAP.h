// evaluation/MAP.h
#pragma once
#include "../core/Retrieval.h"
#include "../core/FeatureDatabase.h"
#include <string>
#include <vector>
#include <functional>
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

// Callback type: nhận queryPath, trả về vector<SearchResult>
using SearchFn = function<vector<SearchResult>(const string&)>;

class MAPEvaluator {
public:
    /**
     * @brief Tách chuỗi trích xuất nhãn phân lớp (Class Label) dựa theo cấu trúc tên đường dẫn
     */
    static string getLabel(const string& path);

    /**
     * @brief Tính toán chỉ số độ chính xác trung bình AP (Average Precision) cho một ảnh mục tiêu tại ngưỡng K.
     */
    static double computeAP(const string& queryPath, const vector<SearchResult>& results, const FeatureDatabase& db, int K);

    /**
     * @brief Tính toán chỉ số MAP@K (Mean Average Precision) trên tập nhiều ảnh query.
     * @param queryPaths   Danh sách đường dẫn các ảnh dùng làm query
     * @param searchFn     Hàm thực hiện tìm kiếm: nhận đường dẫn, trả về Top-K results
     * @param db           FeatureDatabase tham chiếu (để đếm totalRelevant)
     * @param K            Ngưỡng K
     * @param outAPs       (tuỳ chọn) Nếu không nullptr, sẽ được điền danh sách AP từng query
     * @return MAP@K
     */
    static double computeMAP(const vector<string>& queryPaths,
                              const SearchFn& searchFn,
                              const FeatureDatabase& db,
                              int K,
                              vector<double>* outAPs = nullptr);
};