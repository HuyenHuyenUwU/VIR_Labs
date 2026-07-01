// features/Corner.cpp
#include "Corner.h"
#include <algorithm>
using namespace std;

vector<float> CornerFeature::extract(const cv::Mat& image) const {
	// Tiền xử lý ảnh: chuẩn hóa kích thước, lọc nhiễu, chuyển sang ảnh xám
    cv::Mat gray = preprocess(image, true);
	// Phát hiện các góc trong ảnh xám
    vector<cv::Point2f> corners;
	// Hàm GoodFeaturesToTrack phát hiện các góc trong ảnh xám và lưu vào vector corners
    cv::goodFeaturesToTrack(gray, corners, 100, 0.01, 10);

	// Tính histogram không gian 4x4 dựa trên các góc được phát hiện
    vector<float> spatialHist(16, 0.0f);
    for (const auto& pt : corners) {
        int bin_x = std::min(3, static_cast<int>(pt.x / 64.0f));
        int bin_y = std::min(3, static_cast<int>(pt.y / 64.0f));
        spatialHist[bin_x + bin_y * 4] += 1.0f;
    }
    return spatialHist;
}

string CornerFeature::name() const { return "Corner"; }