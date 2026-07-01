// features/Blob.cpp
#include "Blob.h"
#include <algorithm>
using namespace std;

vector<float> BlobFeature::extract(const cv::Mat& image) const {
	// chuyển sang gray scale và làm mờ để giảm nhiễu
    cv::Mat gray = preprocess(image, true);

	// dùng SimpleBlobDetector để phát hiện các blob
    cv::SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 10;
    params.maxArea = 5000;

	// detect blobs
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray, keypoints);

	// tạo histogram với 16 bins dựa trên kích thước của các blob
    vector<float> blobHist(16, 0.0f);
    for (const auto& kp : keypoints) {
        int bin = std::min(15, static_cast<int>(kp.size / 20.0f));
        blobHist[bin] += 1.0f;
    }
    return blobHist;
}

string BlobFeature::name() const { return "Blob"; }