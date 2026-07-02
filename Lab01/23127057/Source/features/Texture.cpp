// features/Texture.cpp
#include "Texture.h"
using namespace std;

vector<float> TextureFeature::extract(const cv::Mat& image) const {
	// Tiền xử lý ảnh: chuẩn hóa kích thước, lọc nhiễu, chuyển sang ảnh xám
    cv::Mat gray = preprocess(image, true);
	// Tính toán Local Binary Pattern (LBP)
	cv::Mat lbp = cv::Mat::zeros(gray.size(), CV_8UC1); // CV_8UC1: 8-bit unsigned single-channel

    // Duyệt từng pixel
    for (int i = 1; i < gray.rows - 1; ++i) {
        for (int j = 1; j < gray.cols - 1; ++j) {
			// tính LBP code cho pixel (i, j)
            uchar center = gray.at<uchar>(i, j);
            uchar code = 0;
            code |= (gray.at<uchar>(i - 1, j - 1) >= center) << 7;
            code |= (gray.at<uchar>(i - 1, j) >= center) << 6;
            code |= (gray.at<uchar>(i - 1, j + 1) >= center) << 5;
            code |= (gray.at<uchar>(i, j + 1) >= center) << 4;
            code |= (gray.at<uchar>(i + 1, j + 1) >= center) << 3;
            code |= (gray.at<uchar>(i + 1, j) >= center) << 2;
            code |= (gray.at<uchar>(i + 1, j - 1) >= center) << 1;
            code |= (gray.at<uchar>(i, j - 1) >= center) << 0;
			// Gán giá trị LBP code vào ma trận lbp
            lbp.at<uchar>(i, j) = code;
        }
    }

	// Tính histogram của LBP
    int histSize = 32;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&lbp, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    vector<float> featureVector;
    hist.copyTo(featureVector);
    return featureVector;
}

string TextureFeature::name() const { return "Texture"; }