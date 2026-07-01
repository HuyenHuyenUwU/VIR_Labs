// indexing/InvertedIndex.h
#pragma once
#include <map>
#include <vector>
#include <string>
using namespace std;

/**
 * @brief Cấu trúc node lưu thông tin file ảnh trong danh sách liên kết của từ vựng
 */
struct IndexNode {
    string imagePath;
    float weight;
};

/**
 * @brief Lớp quản lý ctdl Inverted Index giúp tối ưu hóa và tăng tốc truy vấn
 */
class InvertedIndex {
private:
    map<int, vector<IndexNode>> m_index; /**< Cấu trúc bảng chỉ mục ánh xạ WordID -> List[Images] */

public:
    /**
     * @brief Xóa toàn bộ chỉ mục cũ
     */
    void clear();

    /**
     * @brief Thêm một tài liệu ảnh kèm vector trọng số BoW vào hệ thống Inverted Index
     */
    void addDocument(const string& path, const vector<float>& bowVector);

    /**
     * @brief Tra cứu nhanh danh sách ứng viên có chung từ vựng thị giác nhằm tăng tốc loại biên tìm kiếm
     */
    vector<pair<string, float>> search(const vector<float>& queryBow) const;
};