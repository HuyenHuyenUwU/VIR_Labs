// features/Texture.h
#pragma once
#include "../core/Feature.h"
using namespace std;

/**
 * @brief Lớp trích xuất hoa văn bề mặt dựa vào giải thuật LBP (Texture Feature)
 */

/* @details Giải thuật LBP
* 1. Chuyển ảnh sang ảnh xám
* 2. Duyệt từng pixel (trừ biên) và so sánh với 8 pixel xung quanh
* 3. Nếu pixel xung quanh >= pixel trung tâm, gán giá trị 1, ngược lại gán giá trị 0
* 4. Tính giá trị nhị phân 8 bit từ các pixel xung quanh và gán cho pixel trung tâm 
* VD: Các pixel theo chiều kim đồng hồ từ trên xuống dưới, từ trái sang phải: 1 0 1 1 0 0 1 0 => giá trị nhị phân = 10110010 = 178
* 5. Tính histogram của các giá trị LBP và chuẩn hóa
*/
class TextureFeature : public Feature {
public:
    vector<float> extract(const cv::Mat& image) const override;
    string name() const override;
};