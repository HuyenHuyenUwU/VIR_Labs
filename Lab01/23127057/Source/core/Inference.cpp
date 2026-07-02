// core/Inference.cpp
#include "Inference.h"
using namespace std;

bool InferenceEngine::initialize(const string& basePath) {
    bool l1 = m_blobDB.load(basePath + "db_blob.yml");
    bool l2 = m_colorHistogramDB.load(basePath + "db_colorhistogram.yml");
    bool l3 = m_colorCorrelogramDB.load(basePath + "db_colorcorrelogram.yml");
    bool l4 = m_cornerDB.load(basePath + "db_corner.yml");
    bool l5 = m_siftDB.load(basePath + "db_sift.yml");
    bool l6 = m_orbDB.load(basePath + "db_orb.yml");
    bool l7 = m_textureDB.load(basePath + "db_texture.yml");

    m_isInitialized = (l1 && l2 && l3 && l4 && l5 && l6 && l7);
    return m_isInitialized;
}

InferenceOutput InferenceEngine::query(const cv::Mat& queryImage, const vector<float>& weights, int K) {
    InferenceOutput output;
    output.executionTimeMs = 0.0;

    if (!m_isInitialized || queryImage.empty()) {
        return output;
    }

    // --- BẮT ĐẦU BẤM GIỜ TÌM KIẾM ONLINE ---
    auto startTime = std::chrono::high_resolution_clock::now();

    // 1. Đóng gói danh sách con trỏ đối tượng phục vụ Late Fusion (7 đặc trưng)
    vector<const Feature*> extractors = { 
        &m_blobExtractor, 
        &m_colorHistogramExtractor, 
        &m_colorCorrelogramExtractor,
        &m_cornerExtractor,
        &m_siftExtractor,
        &m_orbExtractor,
        &m_textureExtractor 
    };
    vector<const FeatureDatabase*> dbs = { 
        &m_blobDB, 
        &m_colorHistogramDB, 
        &m_colorCorrelogramDB,
        &m_cornerDB,
        &m_siftDB,
        &m_orbDB,
        &m_textureDB 
    };

    // 2. Gọi hàm lõi xử lý thuật toán kết hợp đặc trưng đã viết ở core/Retrieval
    output.results = Retrieval::searchLateFusion(queryImage, extractors, dbs, weights, K);

    // --- KẾT THÚC BẤM GIỜ ---
    auto endTime = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = endTime - startTime;

    output.executionTimeMs = elapsed.count(); // Lưu lại thời gian phản hồi (ms)
    return output;
}