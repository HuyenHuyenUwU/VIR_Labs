// features/ColorHistogram.cpp
#include "ColorHistogram.h"
using namespace std;

vector<float> ColorHistogram::extract(const cv::Mat& image) const {
    // 1. Tiền xử lý: gọi hàm từ class cha Feature nhưng đặt toGray = false để giữ nguyên màu
    cv::Mat processed = preprocess(image, false);

    if (processed.empty()) {
        return std::vector<float>();
    }

    // 2. Chuyển đổi không gian màu từ BGR sang HSV để hạn chế ảnh hưởng của độ sáng cường độ (V)
    cv::Mat hsv;
    cv::cvtColor(processed, hsv, cv::COLOR_BGR2HSV);

    // 3. Cấu hình các tham số phân chia lược đồ (Binning)
    // Chia kênh Hue làm 8 phần, Saturation làm 8 phần, Value làm 4 phần.
    // Tổng số chiều của vector đặc trưng = 8 * 8 * 4 = 256 chiều.
    int h_bins = 8;
    int s_bins = 8;
    int v_bins = 4;
    int histSize[] = { h_bins, s_bins, v_bins };

    // Định nghĩa dải giá trị chuẩn của các kênh trong OpenCV
    float h_ranges[] = { 0, 180 };  // Kênh H chạy từ 0 đến 179
    float s_ranges[] = { 0, 256 };  // Kênh S chạy từ 0 đến 255
    float v_ranges[] = { 0, 256 };  // Kênh V chạy từ 0 đến 255
    const float* ranges[] = { h_ranges, s_ranges, v_ranges };

    // Lấy thông tin từ cả 3 kênh màu (0: H, 1: S, 2: V)
    int channels[] = { 0, 1, 2 };

    // 4. Tính toán biểu đồ tần suất Histogram bằng hàm tối ưu của OpenCV
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);

    // 5. Chuẩn hóa vector đặc trưng về dải [0, 1] theo chuẩn NORM_MINMAX 
    // Giúp loại bỏ sự chênh lệch lớn về số lượng pixel giữa các vùng ảnh
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    // 6. Chuyển đổi ma trận nhiều chiều (cv::Mat) thành vector 1 chiều (std::vector<float>)
    vector<float> featureVector;

    // Đảm bảo ma trận được duỗi thẳng (flatten) thành 1 hàng trước khi sao chép vùng nhớ
    cv::Mat flatHist = hist.reshape(1, 1);
    flatHist.copyTo(featureVector);

    return featureVector;
}

string ColorHistogram::name() const { return "ColorHistogram"; }