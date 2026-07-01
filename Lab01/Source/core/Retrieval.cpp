// core/Retrieval.cpp
#include "Retrieval.h"
#include <algorithm>
using namespace std;

vector<SearchResult> Retrieval::search(const cv::Mat& queryImage,
    const Feature& extractor,
    const FeatureDatabase& db,
    const InvertedIndex& invIndex,
    int K,
    bool useInvertedIndex)
{
    vector<SearchResult> results;
    if (queryImage.empty()) return results;

    // trích xuất đặc trưng từ ảnh query
    vector<float> queryVec = extractor.extract(queryImage);

	// Đặc trưng là SIFT_BoW hoặc ORB_BoW và bật chế độ tìm kiếm nhanh bằng inverted index
    if (useInvertedIndex && (extractor.name() == "SIFT_BoW" || extractor.name() == "ORB_BoW")) {
        auto candidates = invIndex.search(queryVec);
        for (const auto& c : candidates) {
            results.push_back({ c.first, c.second });
        }
    }
	// Nếu không sử dụng inverted index, thực hiện tìm kiếm tuần tự trên toàn bộ csdl
    else {
		// tính scroe tương đồng giữa vector đặc trưng truy vấn và các vector đặc trưng trong csdl
        for (const auto& [path, vec] : db.getData()) {
            float score = extractor.similarity(queryVec, vec);
            results.push_back({ path, score });
        }
    }

    // Sắp xếp kết quả theo thứ tự giảm dần của độ tương đồng
    sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score;
        });

	// lấy K kết quả đầu tiên nếu số lượng kết quả lớn hơn K
    if (results.size() > static_cast<size_t>(K)) {
        results.resize(K);
    }
    return results;
}

vector<SearchResult> Retrieval::searchLateFusion(const cv::Mat& queryImage,
    const vector<const Feature*>& extractors,
    const vector<const FeatureDatabase*>& databases,
    const vector<float>& weights,
    int K)
{
    vector<SearchResult> finalResults;
    if (queryImage.empty() || extractors.empty() || databases.empty() || extractors.size() != databases.size()) {
        return finalResults;
    }

    // 1. Trích xuất đặc trưng của ảnh Query cho TẤT CẢ các loại đặc trưng được chọn
    vector<vector<float>> queryVectors;
    for (const auto& extractor : extractors) {
        queryVectors.push_back(extractor->extract(queryImage));
    }

    // 2. Lấy danh sách tất cả đường dẫn ảnh từ Database đầu tiên làm chuẩn duyệt
    const auto& primaryData = databases[0]->getData();

    // 3. Duyệt qua từng bức ảnh có trong Csdl
    for (const auto& [imagePath, firstFeatureVec] : primaryData) {
        float combinedScore = 0.0f;

        // Tính điểm tương đồng cho từng loại đặc trưng trên bức ảnh này
        for (size_t i = 0; i < extractors.size(); ++i) {
            // Lấy vector đặc trưng thứ i của bức ảnh hiện tại từ database tương ứng
            const auto& dbData = databases[i]->getData();
            if (dbData.count(imagePath) == 0) continue; // Phòng lỗi nếu ảnh thiếu dữ liệu

            const vector<float>& targetFeatureVec = dbData.at(imagePath);

            // Tính độ tương đồng Cosine
            float currentScore = extractors[i]->similarity(queryVectors[i], targetFeatureVec);

            // Cộng gộp vào điểm tổng số dựa theo trọng số weight tương ứng
            combinedScore += weights[i] * currentScore;
        }

        finalResults.push_back({ imagePath, combinedScore });
    }

    // 4. Sắp xếp lại danh sách kết quả theo điểm tổng hợp giảm dần
    sort(finalResults.begin(), finalResults.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score;
        });

    // Giới hạn số lượng kết quả trả về đúng bằng K
    if (finalResults.size() > static_cast<size_t>(K)) {
        finalResults.resize(K);
    }

    return finalResults;
}