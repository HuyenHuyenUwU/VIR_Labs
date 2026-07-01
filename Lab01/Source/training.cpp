// train.cpp
#include "core/FeatureDatabase.h"
#include "features/Blob.h"
#include "features/ColorHistogram.h"
#include "features/ColorCorrelogram.h"
#include "features/Corner.h"
#include "features/SIFT.h"
#include "features/ORB.h"
#include "features/Texture.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
using namespace std;

namespace fs = std::filesystem;

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

/* Nhiệm vụ: 
File này chứa một hàm main() độc lập (ko liên quan gì tới hàm main() bên main.cpp
Chỉ cần chạy file này đúng 1 lần duy nhất khi có dữ liệu ảnh gốc mới
Nó quét qua toàn bộ thư mục ảnh, trích xuất đặc trưng và đóng gói thành file.yml
Kết quả đầu ra là 2 file: db_color.yml và db_texture.yml, lưu vào thư mục gốc của dự án
(nếu thư mục gốc có 2 file yml này r thì thôi, chuyển sang main.cpp để chạy tìm kiếm)
*/
int main() {
    cout << "=== GIAI DOAN OFFLINE: TRICH XUAT VA DONG GOI HE THONG ===" << endl;

    // 1. Cấu hình đường dẫn thư mục chứa ảnh gốc (Thay đổi theo máy của bạn)
    // Dấu * dùng để lấy tất cả các file ảnh định dạng .jpg, .png, ...
    // Khi bấm F5, ctrinh lấy directory tại thư mục gốc (chứa source/, dataset/)
	// Cách làm: properties -> Debugging -> Working Directory -> chỉnh từ $(ProjectDir) sang $(SolutionDir)..\

    string datasetPattern = "Dataset/train/*.jpg";
    vector<cv::String> imagePaths;
	// 1.1 Sử dụng OpenCV glob để quét tất cả các file ảnh trong thư mục Dataset
    cv::glob(datasetPattern, imagePaths);
        
    if (imagePaths.empty()) {
        cerr << "Loi: Khong tim thay anh nao trong thu muc Dataset!" << endl;
        return -1;
    }

	// 2. Khởi tạo các bộ trích xuất và Database riêng biệt (Mô hình Late Fusion)
	// Mỗi loại đặc trưng sẽ có một database riêng, lưu trữ trên RAM trước khi ghi xuống ổ cứng
	// ta xét 7 loại đặc trưng: blob, color histogram, color correlogram, corner, sift, orb, texture
	BlobFeature blobExtractor;
	ColorHistogram colorHistogramExtractor;
	ColorCorrelogram colorCorrelogramExtractor;
	CornerFeature cornerExtractor;
	SIFTFeature siftExtractor;
	ORBFeature orbExtractor;
	TextureFeature textureExtractor;

	FeatureDatabase blobDB;
	FeatureDatabase colorHistogramDB;
	FeatureDatabase colorCorrelogramDB;
	FeatureDatabase cornerDB;
	FeatureDatabase siftDB;
	FeatureDatabase orbDB;
	FeatureDatabase textureDB;

    cout << "Dang tien hanh trich xuat " << imagePaths.size() << " anh..." << endl;

    size_t processedCount = 0;
    size_t skippedCount = 0;

    // 3. Vòng lặp quét và trích xuất thực tế
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        string path = imagePaths[i];
        cout << "Dang xu ly: " << (i + 1) << "/" << imagePaths.size() << " - " << path << endl;

        cv::Mat img = cv::imread(path);

        if (img.empty()) {
            cout << "Bo qua anh loi: " << path << endl;
            ++skippedCount;
            continue;
        }

        try {
            // Trích xuất các đặc trưng độc lập
            vector<float> blobVec = blobExtractor.extract(img);
            vector<float> colorHistogramVec = colorHistogramExtractor.extract(img);
            vector<float> colorCorrelogramVec = colorCorrelogramExtractor.extract(img);
            vector<float> cornerVec = cornerExtractor.extract(img);
            vector<float> siftVec = siftExtractor.extract(img);
            vector<float> orbVec = orbExtractor.extract(img);
            vector<float> textureVec = textureExtractor.extract(img);

            // Đẩy vào database trên RAM
            blobDB.addEntry(path, blobVec);
            colorHistogramDB.addEntry(path, colorHistogramVec);
            colorCorrelogramDB.addEntry(path, colorCorrelogramVec);
            cornerDB.addEntry(path, cornerVec);
            siftDB.addEntry(path, siftVec);
            orbDB.addEntry(path, orbVec);
            textureDB.addEntry(path, textureVec);
            ++processedCount;
        }
        catch (const cv::Exception& e) {
            cout << "Loi OpenCV khi xu ly anh: " << path << endl;
            cout << e.what() << endl;
            ++skippedCount;
            continue;
        }
        catch (const std::exception& e) {
            cout << "Loi std::exception khi xu ly anh: " << path << endl;
            cout << e.what() << endl;
            ++skippedCount;
            continue;
        }

        if ((i + 1) % 10 == 0 || i == imagePaths.size() - 1) {
            cout << "Da xu ly hop le: " << processedCount << "/" << imagePaths.size()
                 << " anh, bo qua: " << skippedCount << endl;
        }
    }

    // 4. Ghi dữ liệu từ RAM xuống ổ cứng thành file cấu hình
    cout << "Dang luu CSDL xuong o cung..." << endl;
    fs::path featureDatabaseDir = resolveFeatureDatabaseDir();
    fs::create_directories(featureDatabaseDir);
    bool s1 = blobDB.save((featureDatabaseDir / "db_blob.yml").generic_string());
    bool s2 = colorHistogramDB.save((featureDatabaseDir / "db_colorhistogram.yml").generic_string());
    bool s3 = colorCorrelogramDB.save((featureDatabaseDir / "db_colorcorrelogram.yml").generic_string());
    bool s4 = cornerDB.save((featureDatabaseDir / "db_corner.yml").generic_string());
    bool s5 = siftDB.save((featureDatabaseDir / "db_sift.yml").generic_string());
    bool s6 = orbDB.save((featureDatabaseDir / "db_orb.yml").generic_string());
    bool s7 = textureDB.save((featureDatabaseDir / "db_texture.yml").generic_string());

    if (s1 && s2 && s3 && s4 && s5 && s6 && s7) {
        cout << "===> THANH CONG! Da tao ra 7 file database:" << endl;
        cout << "     - " << (featureDatabaseDir / "db_blob.yml").generic_string() << endl;
        cout << "     - " << (featureDatabaseDir / "db_colorhistogram.yml").generic_string() << endl;
        cout << "     - " << (featureDatabaseDir / "db_colorcorrelogram.yml").generic_string() << endl;
        cout << "     - " << (featureDatabaseDir / "db_corner.yml").generic_string() << endl;
        cout << "     - " << (featureDatabaseDir / "db_sift.yml").generic_string() << endl;  
        cout << "     - " << (featureDatabaseDir / "db_orb.yml").generic_string() << endl;
        cout << "     - " << (featureDatabaseDir / "db_texture.yml").generic_string() << endl;
    }
    else {
        cerr << "Loi: Khong the ghi file CSDL xuống o cung!" << endl;
    }

    return 0;
}