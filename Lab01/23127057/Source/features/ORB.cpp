// features/ORB.cpp
#include "ORB.h"
using namespace std;

ORBFeature::ORBFeature() {
    bow.createDummyVocabulary(32); // ORB mặc định 32 chiều nhị phân
}

vector<float> ORBFeature::extract(const cv::Mat& image) const {
	// Tiền xử lý ảnh: chuyển sang ảnh xám và chuẩn hóa kích thước
    cv::Mat gray = preprocess(image, true);

	// Sử dụng ORB để phát hiện keypoints và trích xuất descriptors
    cv::Ptr<cv::ORB> detector = cv::ORB::create(300);
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    return bow.quantize(descriptors);
}

string ORBFeature::name() const { return "ORB"; }