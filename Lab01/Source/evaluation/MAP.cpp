// evaluation/MAP.cpp
#include "MAP.h"
#include <algorithm>
using namespace std;

string MAPEvaluator::getLabel(const string& path) {
    string clean = path;
    replace(clean.begin(), clean.end(), '\\', '/');

    // 1. Lấy tên file (vd: "01_1.jpg" từ "Dataset/train/01_1.jpg")
    size_t last_slash = clean.find_last_of('/');
    string filename = (last_slash == string::npos) ? clean : clean.substr(last_slash + 1);

    // 2. Tách chuỗi trước dấu '_' để làm nhãn phân lớp (vd: "01")
    size_t underscore = filename.find('_');
    if (underscore != string::npos) {
        return filename.substr(0, underscore);
    }

    // Dự phòng: nếu không có dấu '_', lấy thư mục cha theo code cũ
    if (last_slash != string::npos) {
        size_t second_slash = clean.find_last_of('/', last_slash - 1);
        if (second_slash != string::npos) {
            return clean.substr(second_slash + 1, last_slash - second_slash - 1);
        }
    }

    return "unknown";
}

//string MAPEvaluator::getLabel(const string& path) {
//    string clean = path;
//	// xóa các ký tự không hợp lệ và chuẩn hóa đường dẫn
//    // VD: đường dẫn chuẩn có dạng:
//	// .../class_label/image_name.jpg
//	// ta lấy nhãn phân lớp là 'class_label'
//    replace(clean.begin(), clean.end(), '\\', '/');
//    size_t last_slash = clean.find_last_of('/');
//    if (last_slash == string::npos) return "unknown";
//    size_t second_slash = clean.find_last_of('/', last_slash- 1);
//    if (second_slash == string::npos) return "unknown";
//    return clean.substr(second_slash + 1, last_slash - second_slash - 1);
//}

double MAPEvaluator::computeAP(const string& queryPath, const vector<SearchResult>& results, const FeatureDatabase& db, int K) {
    // lấy label
    string qLabel = getLabel(queryPath);

	// đếm số lượng ảnh có nhãn giống ảnh truy vấn
    int totalRelevantInDb = 0;
    for (const auto& [path, vec] : db.getData()) {
        if (getLabel(path) == qLabel) totalRelevantInDb++;
    }
    if (totalRelevantInDb == 0) return 0.0;

	// tính toán AP
	// AP = (1 / min(totalRelevantInDb, K)) * sum(precision@i) với i là vị trí của ảnh đúng nhãn trong top K
    double sumPrecision = 0.0;
    int relevantFound = 0;
    int limit = min(K, static_cast<int>(results.size()));

	// duyệt qua các kết quả tìm kiếm và tính toán độ chính xác tại mỗi vị trí
    for (int i = 0; i < limit; ++i) {
        if (getLabel(results[i].imagePath) == qLabel) {
            relevantFound++;
            sumPrecision += static_cast<double>(relevantFound) / (i + 1);
        }
    }
	// trả về giá trị AP
    return sumPrecision / min(totalRelevantInDb, K);
}

double MAPEvaluator::computeMAP(const vector<string>& queryPaths,
                                 const SearchFn& searchFn,
                                 const FeatureDatabase& db,
                                 int K,
                                 vector<double>* outAPs) {
    if (queryPaths.empty()) return 0.0;

    if (outAPs) {
        outAPs->clear();
        outAPs->reserve(queryPaths.size());
    }

    double sumAP = 0.0;
    for (const auto& qPath : queryPaths) {
        // Thực hiện tìm kiếm cho ảnh query này
        vector<SearchResult> results = searchFn(qPath);

        // Tính AP cho ảnh query này
        double ap = computeAP(qPath, results, db, K);
        sumAP += ap;

        if (outAPs) outAPs->push_back(ap);
    }

    // MAP = trung bình cộng các AP
    return sumAP / static_cast<double>(queryPaths.size());
}