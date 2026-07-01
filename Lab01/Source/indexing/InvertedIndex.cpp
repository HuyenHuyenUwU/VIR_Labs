// indexing/InvertedIndex.cpp
#include "InvertedIndex.h"
using namespace std;

void InvertedIndex::clear() {
    m_index.clear();
}

void InvertedIndex::addDocument(const string& path, const vector<float>& bowVector) {
	// thêm từng từ thứ wordId vào bảng chỉ mục, ánh xạ đến đường dẫn ảnh và trọng số BoW
    for (size_t wordId = 0; wordId < bowVector.size(); ++wordId) {
        if (bowVector[wordId] > 0.0f) {
            m_index[static_cast<int>(wordId)].push_back({ path, bowVector[wordId] });
        }
    }
}

vector<pair<string, float>> InvertedIndex::search(const vector<float>& queryBow) const {
	// accumulators lưu trữ tổng trọng số BoW cho mỗi ảnh ứng viên
    map<string, float> accumulators;

    for (size_t wordId = 0; wordId < queryBow.size(); ++wordId) {
        float qWeight = queryBow[wordId];
        if (qWeight > 0.0f && m_index.count(static_cast<int>(wordId))) {
            const auto& postings = m_index.at(static_cast<int>(wordId));
            for (const auto& node : postings) {
				// Cộng dồn trọng số BoW của ảnh ứng viên
                accumulators[node.imagePath] += qWeight * node.weight;
            }
        }
    }
    return vector<pair<string, float>>(accumulators.begin(), accumulators.end());
}