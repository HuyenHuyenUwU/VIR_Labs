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
            // queryWithAP: tìm kiếm + tính AP@K cho query này
            return engine.queryWithAP(queryPath, weights, K);
        }
    );

    // Vòng lặp chính
    gui.run();

    cout << "Goodbye!" << endl;
    return 0;
}
