// main.cpp
#include "core/Inference.h"
#include "gui/gui.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
using namespace std;

// Điểm khởi động chính của hệ thống CBIR
// Kết nối InferenceEngine (backend) với CBIRGui (giao diện) qua SearchCallback
//
// Cách chạy:
//   1. Chạy training.cpp trước để sinh 7 file db_*.yml trong thư mục FeatureDatabase/ (nếu có rồi thì thôi)
//   2. Chạy main.cpp này để mở giao diện tìm kiếm
//
// Phím tắt trong giao diện:
//   Enter  => Tìm kiếm
//   ESC/Q  => Thoát

namespace fs = filesystem;

static fs::path resolveFeatureDatabaseDir() {
    fs::path current = fs::current_path();
    for (int i = 0; i < 6; ++i) {
        fs::path candidate = current / "FeatureDatabase";
        if (fs::exists(candidate) && fs::is_directory(candidate)) {
            return candidate;
        }
        if (!current.has_parent_path()) {
            break;
        }
        current = current.parent_path();
    }
    return fs::current_path() / "FeatureDatabase";
}

// Trọng số các feature
float BLOB_W = 0.08f; 
float COLORHISTOGRAM_W = 0.14f;
float COLORCORRELOGRAM_W = 0.14f;
float CORNER_W = 0.08f;
float SIFT_W = 0.14f;
float ORB_W = 0.14f;
float TEXTURE_W = 0.28f; // trọng số cao nhất vì Texture thường hiệu quả nhất

// Hàm bổ trợ: Ghép các bức ảnh kết quả thành một lưới ảnh (Grid) trực quan đẹp mắt để hiển thị lên màn hình đồ họa
//void displayResultGrid(const cv::Mat& queryImg, const vector<SearchResult>& results, double timeMs) {
//    int thumbSize = 150; // Mỗi ảnh kết quả sẽ thu nhỏ về kích thước vuông 150x150
//    int cols = 5;        // Hiển thị tối đa 5 ảnh trên một hàng
//    int rows = (static_cast<int>(results.size()) + cols - 1) / cols;
//
//    // Tạo một canvas màu xám lớn làm nền
//    cv::Mat canvas = cv::Mat::zeros(rows * thumbSize + 100, cols * thumbSize, CV_8UC3);
//    canvas.setTo(cv::Scalar(50, 50, 50)); // Màu nền xám tối thanh lịch
//
//    // Vẽ ảnh Query lên góc trái phía trên canvas để so sánh
//    cv::Mat queryResized;
//    cv::resize(queryImg, queryResized, cv::Size(80, 80));
//    queryResized.copyTo(canvas(cv::Rect(10, 10, 80, 80)));
//
//    // Chèn thông số thời gian phản hồi của hệ thống lên giao diện đồ họa
//    string infoText = "Query Time: " + to_string(timeMs) + " ms | Total Results: " + to_string(results.size());
//    cv::putText(canvas, infoText, cv::Point(110, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
//
//    // Tiến hành lặp để vẽ lưới ảnh kết quả Top K
//    for (size_t i = 0; i < results.size(); ++i) {
//        cv::Mat img = cv::imread(results[i].imagePath);
//        if (img.empty()) continue;
//
//        cv::resize(img, img, cv::Size(thumbSize - 10, thumbSize - 10)); // Thu nhỏ kèm viền cách nhau
//
//        int r = static_cast<int>(i) / cols;
//        int c = static_cast<int>(i) % cols;
//        int x = c * thumbSize + 5;
//        int y = r * thumbSize + 100 + 5; // Dịch xuống 100 pixel chừa không gian cho thanh thông số tiêu đề
//
//        img.copyTo(canvas(cv::Rect(x, y, thumbSize - 10, thumbSize - 10)));
//
//        // Ghi điểm số tương đồng (Score) lên ngay phía dưới mỗi bức ảnh nhỏ
//        string scoreStr = "S: " + to_string(results[i].score).substr(0, 4);
//        cv::putText(canvas, scoreStr, cv::Point(x + 5, y + thumbSize - 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
//    }
//
//    cv::imshow("CBIR System - Search Results", canvas);
//    cv::waitKey(0);
//}

int main() {
    cout << "=== GIAI DOAN ONLINE: KHOI DONG HE THONG TIM KIEM ===" << endl;

    InferenceEngine engine;
    // Nạp dữ liệu đặc trưng vào RAM (7 đặc trưng: blob, colorhistogram, colorcorrelogram, corner, sift, orb, texture)
    fs::path featureDatabaseDir = resolveFeatureDatabaseDir();
    if (!engine.initialize(featureDatabaseDir.generic_string() + "/")) {
        cerr << "Loi: Khong tim thay hoac khong nạp duoc cac file db_*.yml!" << endl;
        cerr << "Vui long chay file training.cpp truoc de tao ra 7 file database." << endl;
        return -1;
    }
    cout << "He thong da san sang nhan lenh truy van (7 dac trung)!" << endl;
    // Tạo GUI
    CBIRGui gui("CSC16105 — CBIR System");

    // Kết nối engine vào GUI qua SearchCallback
    // Lambda capture engine by reference - engine sống suốt vòng lặp GUI
    gui.setSearchCallback(
        [&engine](const string& queryPath,
            const vector<float>& weights,
            int K) -> InferenceOutput
        {
            cv::Mat img = cv::imread(queryPath);
            if (img.empty()) {
                cerr << "[WARN] Cannot read query image: " << queryPath << "\n";
                return InferenceOutput{};
            }
            return engine.query(img, weights, K);
        }
    );

    // Vòng lặp chính
    gui.run();

    cout << "Goodbye!" << endl;
    return 0;
}
