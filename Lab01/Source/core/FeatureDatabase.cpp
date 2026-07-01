// core/FeatureDatabase.cpp
#include "FeatureDatabase.h"
using namespace std;

void FeatureDatabase::addEntry(const string& path, const vector<float>& vec) {
	// nếu path đã tồn tại thì ghi đè, nếu chưa tồn tại thì thêm mới
    m_db[path] = vec;
}

const map<string, vector<float>>& FeatureDatabase::getData() const {
	// trả về tham chiếu hằng đến map dữ liệu đặc trưng
    return m_db;
}

void FeatureDatabase::clear() {
	// xóa toàn bộ dữ liệu trong map
    m_db.clear();
}

bool FeatureDatabase::save(const string& filepath) const {
	// tạo một đối tượng FileStorage để ghi dữ liệu vào file .yaml
	cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
	if (!fs.isOpened()) return false;
	// lưu số lượng mục dữ liệu và danh sách các mục dữ liệu vào file
	fs << "DatabaseSize" << static_cast<int>(m_db.size());
	fs << "Entries" << "["; // start sequence

	for (const auto& kv : m_db) {
		// Each entry is a map (block style)
		fs << "{"; // start map
		fs << "path" << kv.first;
		cv::Mat mat(kv.second); // convert vector<float> -> Mat
		fs << "vector" << mat;
		fs << "}"; // end map
	}

	fs << "]"; // end sequence
	fs.release();
	return true;
}


bool FeatureDatabase::load(const string& filepath) {
    try {
        // tạo một đối tượng FileStorage để đọc dữ liệu từ file .yaml
        cv::FileStorage fs(filepath, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;
        m_db.clear();

        // đọc số lượng mục dữ liệu từ file
        cv::FileNode entries = fs["Entries"];
        if (entries.type() != cv::FileNode::SEQ) {
            fs.release();
            return false;
        }
        // đọc từng mục dữ liệu và lưu vào map m_db
        for (cv::FileNodeIterator it = entries.begin(); it != entries.end(); ++it) {
            string path;
            (*it)["path"] >> path;
            cv::Mat matVec;
            (*it)["vector"] >> matVec;
            vector<float> vec;
            if (!matVec.empty())
                matVec.copyTo(vec);
            m_db[path] = vec;
        }
        fs.release();
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "FileStorage parse error: " << e.what() << std::endl;
        return false;
    }
}