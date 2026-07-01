// core/FeatureDatabase.h
#pragma once
#include <map>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

/**
 * @brief Class quản lý việc lưu trữ, đọc và ghi csdl đặc trưng dưới dạng YAML
 * @details Csdl đặc trưng bao gồm các cặp key-value, trong đó key là đường dẫn ảnh và value là vector đặc trưng tương ứng
 * @details Class này cung cấp các hàm để thêm, sửa, xóa, đọc/ghi file YAML, và truy xuất dữ liệu đặc trưng từ bộ nhớ RAM
 */
class FeatureDatabase {
private:
    map<string, vector<float>> m_db; /**< Bản đồ ánh xạ từ đường dẫn ảnh đến vector đặc trưng */

public:
    /**
     * @brief Thêm một mục dữ liệu đặc trưng mới vào bộ nhớ RAM.
     */
    void addEntry(const string& path, const vector<float>& vec);

    /**
     * @brief Lấy toàn bộ danh sách dữ liệu đặc trưng hiện có.
     */
    const map<string, vector<float>>& getData() const;

    /**
     * @brief Xóa toàn bộ dữ liệu trong database.
     */
    void clear();

    /**
     * @brief Lưu cấu trúc dữ liệu đặc trưng từ RAM xuống file cấu hình .yml[cite: 47].
     */
    bool save(const string& filepath) const;

    /**
     * @brief Đọc dữ liệu đặc trưng từ file cấu hình .yml lên RAM phục vụ so khớp[cite: 51].
     */
    bool load(const string& filepath);
};