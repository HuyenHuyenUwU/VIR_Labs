// gui/GUI.h
// Giao diện đồ họa cho hệ thống CBIR (Content-Based Image Retrieval)
// Dùng OpenCV highgui + mouse callback

#pragma once
#include "../core/Inference.h"  // Lấy InferenceOutput từ đây
#include "../core/Retrieval.h"   // Lấy SearchResult từ đây
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional>
#include <array>
using namespace std;

// Callback: được gọi khi người dùng nhấn "Search" 
using SearchCallback = function<InferenceOutput(
    const string&,         // đường dẫn ảnh query
    const vector<float>&,  // trọng số 7 đặc trưng (sẽ được normalize)
    int                    // số ảnh trả về K
)>;

// Tên 7 đặc trưng (thứ tự khớp với weights[] trong main và Inference)
static const array<string, 7> FEATURE_NAMES = {
    "Blob", "ColorHistogram", "ColorCorrelogram", "Corner", "SIFT", "ORB", "Texture"
};

// ============================================================================
//  CBIRGui — Toàn bộ giao diện chỉ 1 cửa sổ OpenCV
// ============================================================================
class CBIRGui {
public:
    // bố cục 
    static constexpr int WIN_W = 1280;
    static constexpr int WIN_H = 840;
    static constexpr int PANEL_W = 300;
    static constexpr int RESULT_W = WIN_W - PANEL_W;
    static constexpr int THUMB_SIZE = 155;
    static constexpr int CTRL_X = 10;
    static constexpr int CTRL_W = PANEL_W - 20;

    // Màu sắc 
    static const cv::Scalar BG_DARK;
    static const cv::Scalar BG_PANEL;
    static const cv::Scalar ACCENT;
    static const cv::Scalar TEXT_PRIMARY;
    static const cv::Scalar TEXT_DIM;
    static const cv::Scalar BTN_NORMAL;
    static const cv::Scalar BTN_ACTIVE;
    static const cv::Scalar SLIDER_BG;
    static const cv::Scalar SLIDER_FILL;
    static const cv::Scalar BORDER;

    // Constructor, Destructor
    explicit CBIRGui(const string& windowTitle = "CBIR System");
    ~CBIRGui();

    // API công khai
    void setSearchCallback(SearchCallback cb);
    void run(); // blocking event loop

private:
    // State
    string                   m_windowTitle;
    cv::Mat                  m_canvas;
    SearchCallback           m_searchCb;

    string                   m_queryPath;
    cv::Mat                  m_queryImg;
    vector<SearchResult>     m_results;
    double                   m_queryTimeMs = 0.0;

    array<float, 7>          m_weights = { 
        0.08f, 0.14f, 0.14f, 0.08f, 0.14f, 0.14f, 0.28f
    };
    int                      m_K = 10;
    int                      m_scrollRow = 0;
    int                      m_hoveredThumb = -1;

    // Slider drag state
    int   m_activeSlider = -1;
    bool  m_dragging = false;

    // Hit-areas (recalculated each frame in drawPanel)
    cv::Rect m_browseRect;
    cv::Rect m_searchRect;
    cv::Rect m_resetRect;
    cv::Rect m_kMinusRect;
    cv::Rect m_kPlusRect;

    struct SliderArea {
        cv::Rect  track;
        int       featureIdx;
    };
    vector<SliderArea> m_sliderAreas;

    // Render 
    void render();
    void drawPanel();
    void drawResultArea();
    void drawQueryPreview();
    void drawWeightBars(int x, int y, int w);
    void drawResultGrid(int x, int y, int w, int h);
    void drawThumb(const cv::Mat& img, int x, int y, float score,
        int rank, bool hovered, const string& path);
    void drawStatusBar();

    // Widget helpers
    void drawButton(const cv::Rect& r, const string& label,
        bool active = false);
    void drawSlider(int featureIdx, int yTop);
    void drawSectionLabel(const string& text, int yTop);
    void drawProgressBar(int x, int y, int w, int h,
        float val, const cv::Scalar& fill);
    void drawKControl(int yTop);
    void showOverlayMsg(const string& msg);

    // Mouse 
    static void onMouse(int event, int x, int y, int flags, void* ud);
    void        handleMouse(int event, int x, int y, int flags);

    // Helpers
    void        doSearch();
    string      openFileDialog();
    void        normalizeWeights();
    bool        hit(const cv::Rect& r, int x, int y) const;
    float       clampW(float v) const { return max(0.f, min(1.f, v)); }

    static string      fmtFloat(float v, int p = 2);
    static string      fmtTime(double ms);
    static string      shortenPath(const string& p, size_t maxLen);
    static cv::Scalar  scoreColor(float score);
    static void        textCentered(cv::Mat& m, const string& t,
        const cv::Rect& r, double scale,
        const cv::Scalar& color, int thick = 1);
};