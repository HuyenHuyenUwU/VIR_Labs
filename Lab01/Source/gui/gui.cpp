// gui/GUI.cpp
// Mỗi frame render toàn bộ lên m_canvas rồi imshow()

#include "GUI.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
using namespace std;

// Màu sắc
// nền than + nhấn vàng-amber
const cv::Scalar CBIRGui::BG_DARK = { 28,  28,  32 };
const cv::Scalar CBIRGui::BG_PANEL = { 36,  36,  44 };
const cv::Scalar CBIRGui::ACCENT = { 50, 165, 220 };  // BGR → cam nhạt
const cv::Scalar CBIRGui::TEXT_PRIMARY = { 225, 225, 230 };
const cv::Scalar CBIRGui::TEXT_DIM = { 110, 110, 120 };
const cv::Scalar CBIRGui::BTN_NORMAL = { 58,  58,  72 };
const cv::Scalar CBIRGui::BTN_ACTIVE = { 40, 140, 200 };
const cv::Scalar CBIRGui::SLIDER_BG = { 52,  52,  64 };
const cv::Scalar CBIRGui::SLIDER_FILL = { 55, 165, 215 };
const cv::Scalar CBIRGui::BORDER = { 62,  62,  78 };

//Static helpers
string CBIRGui::fmtFloat(float v, int p) {
    ostringstream ss;
    ss << fixed << setprecision(p) << v;
    return ss.str();
}
string CBIRGui::fmtTime(double ms) {
    ostringstream ss;
    ss << fixed << setprecision(1) << ms << " ms";
    return ss.str();
}
string CBIRGui::shortenPath(const string& p, size_t maxLen) {
    if (p.size() <= maxLen) return p;
    return "..." + p.substr(p.size() - (maxLen - 3));
}
cv::Scalar CBIRGui::scoreColor(float score) {
    // Gradient xanh (score cao) => đỏ (score thấp)
    float s = std::max(0.f, min(1.f, score));
    return cv::Scalar(40, static_cast<int>(220 * s), static_cast<int>(220 * (1.f - s)));
}
void CBIRGui::textCentered(cv::Mat& m, const string& t, const cv::Rect& r,
    double sc, const cv::Scalar& col, int thick) {
    int base = 0;
    auto sz = cv::getTextSize(t, cv::FONT_HERSHEY_SIMPLEX, sc, thick, &base);
    int tx = r.x + (r.width - sz.width) / 2;
    int ty = r.y + (r.height + sz.height) / 2 - base / 2;
    cv::putText(m, t, { tx, ty }, cv::FONT_HERSHEY_SIMPLEX, sc, col, thick, cv::LINE_AA);
}
bool CBIRGui::hit(const cv::Rect& r, int x, int y) const {
    if (x < r.x || x >= r.x + r.width) return false;
    if (y < r.y || y >= r.y + r.height) return false;
    return true;
}

//  Constructor / Destructor
CBIRGui::CBIRGui(const string& title)
    : m_windowTitle(title)
    , m_canvas(WIN_H, WIN_W, CV_8UC3)
{
    cv::namedWindow(m_windowTitle, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(m_windowTitle, onMouse, this);
}
CBIRGui::~CBIRGui() {
    cv::destroyWindow(m_windowTitle);
}

void CBIRGui::setSearchCallback(SearchCallback cb) {
    m_searchCb = std::move(cb);
}

// ============================================================================
//  Event loop
// ============================================================================
void CBIRGui::run() {
    render();
    cv::imshow(m_windowTitle, m_canvas);
    while (true) {
        int key = cv::waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q') break;
        if (key == 13) { doSearch(); continue; } // Enter → search
        render();
        cv::imshow(m_windowTitle, m_canvas);
    }
}

// ============================================================================
//  Render tổng thể
// ============================================================================
void CBIRGui::render() {
    m_canvas.setTo(BG_DARK);

    // Panel trái
    cv::rectangle(m_canvas, { 0, 0, PANEL_W, WIN_H }, BG_PANEL, cv::FILLED);
    cv::line(m_canvas, { PANEL_W, 0 }, { PANEL_W, WIN_H }, BORDER, 1);

    drawPanel();
    drawResultArea();
    drawMAPPanel();
    drawStatusBar();
}

// ============================================================================
//  Panel trái
// ============================================================================
void CBIRGui::drawPanel() {
    m_sliderAreas.clear();

    int y = 14;

    // Logo / tiêu đề
    cv::putText(m_canvas, "CBIR System",
        { CTRL_X, y + 20 }, cv::FONT_HERSHEY_SIMPLEX,
        0.82, ACCENT, 2, cv::LINE_AA);
    y += 30;
    cv::putText(m_canvas, "Content-Based Image Retrieval",
        { CTRL_X, y + 12 }, cv::FONT_HERSHEY_SIMPLEX,
        0.42, TEXT_DIM, 1, cv::LINE_AA);
    y += 20;
    cv::line(m_canvas, { CTRL_X, y }, { CTRL_X + CTRL_W, y }, BORDER, 1);
    y += 14;

    //Section 1: Query Image 
    drawSectionLabel("1  Query Image", y);
    y += 24;

    m_browseRect = { CTRL_X, y, CTRL_W, 30 };
    drawButton(m_browseRect, "Browse File ...");
    y += 36;

    string dispPath = m_queryPath.empty()
        ? "(no file selected)"
        : shortenPath(m_queryPath, 34);
    cv::putText(m_canvas, dispPath,
        { CTRL_X, y + 12 }, cv::FONT_HERSHEY_SIMPLEX,
        0.40, m_queryPath.empty() ? TEXT_DIM : TEXT_PRIMARY,
        1, cv::LINE_AA);
    y += 22;

    // Section 2: Top-K 
    y += 6;
    drawSectionLabel("2  Top-K Results", y);
    y += 24;
    drawKControl(y);
    y += 38;

    // Section 3: Feature Weights
    y += 6;
    drawSectionLabel("3  Feature Weights", y);
    y += 24;

    for (int i = 0; i < 7; ++i) {
        drawSlider(i, y);
        y += 48;
    }

    // Reset + Normalize
    m_resetRect = { CTRL_X, y, (CTRL_W - 6) / 2, 26 };
    drawButton(m_resetRect, "Reset");

    cv::Rect normRect = { CTRL_X + (CTRL_W - 6) / 2 + 6, y, (CTRL_W - 6) / 2, 26 };
    drawButton(normRect, "Normalize");
    // Lưu normRect để xử lý click trong mouse handler
    // (dùng tag gắn vào m_resetRect + 1 trick: check x offset)
    y += 32;

    // Nút SEARCH
    y += 8;
    m_searchRect = { CTRL_X, y, CTRL_W, 36 };
    bool canSearch = !m_queryPath.empty();
    // Vẽ nền nút nổi bật nếu có thể search
    if (canSearch) {
        cv::rectangle(m_canvas, m_searchRect, BTN_ACTIVE, cv::FILLED);
        cv::rectangle(m_canvas, m_searchRect, ACCENT, 1);
        textCentered(m_canvas, "  SEARCH", m_searchRect, 0.55,
            { 15, 15, 15 }, 2);
    }
    else {
        cv::rectangle(m_canvas, m_searchRect, BTN_NORMAL, cv::FILLED);
        cv::rectangle(m_canvas, m_searchRect, BORDER, 1);
        textCentered(m_canvas, "  SEARCH", m_searchRect, 0.55, TEXT_DIM, 1);
    }
    y += 44;

    // Phím tắt hint
    cv::putText(m_canvas, "Enter = Search   ESC = Quit",
        { CTRL_X, y + 12 }, cv::FONT_HERSHEY_SIMPLEX,
        0.40, TEXT_DIM, 1, cv::LINE_AA);
}

// Section label 
void CBIRGui::drawSectionLabel(const string& text, int yTop) {
    cv::rectangle(m_canvas, { CTRL_X - 4, yTop, CTRL_W + 8, 20 },
        { 48, 48, 60 }, cv::FILLED);
    cv::rectangle(m_canvas, { CTRL_X - 4, yTop, 3, 20 }, ACCENT, cv::FILLED);
    cv::putText(m_canvas, text, { CTRL_X + 5, yTop + 14 },
        cv::FONT_HERSHEY_SIMPLEX, 0.42, ACCENT, 1, cv::LINE_AA);
}

// Button
void CBIRGui::drawButton(const cv::Rect& r, const string& label, bool active) {
    cv::rectangle(m_canvas, r, active ? BTN_ACTIVE : BTN_NORMAL, cv::FILLED);
    cv::rectangle(m_canvas, r, active ? ACCENT : BORDER, 1);
    textCentered(m_canvas, label, r, 0.45,
        active ? cv::Scalar{ 15,15,15 } : TEXT_PRIMARY,
        active ? 2 : 1);
}

// K control
void CBIRGui::drawKControl(int yTop) {
    m_kMinusRect = { CTRL_X, yTop, 32, 28 };
    m_kPlusRect = { CTRL_X + CTRL_W - 32, yTop, 32, 28 };
    drawButton(m_kMinusRect, "-");
    drawButton(m_kPlusRect, "+");
    cv::Rect lbl = { CTRL_X + 36, yTop, CTRL_W - 72, 28 };
    textCentered(m_canvas, "K = " + std::to_string(m_K), lbl,
        0.50, TEXT_PRIMARY, 1);
}

// Slider 
void CBIRGui::drawSlider(int fi, int yTop) {
    float val = m_weights[fi];

    // Label + value
    string lbl = FEATURE_NAMES[fi] + "  " + fmtFloat(val, 2);
    cv::putText(m_canvas, lbl, { CTRL_X, yTop + 13 },
        cv::FONT_HERSHEY_SIMPLEX, 0.42, TEXT_PRIMARY, 1, cv::LINE_AA);

    // Track
    int ty = yTop + 22;
    cv::Rect track = { CTRL_X, ty, CTRL_W, 6 };
    cv::rectangle(m_canvas, track, SLIDER_BG, cv::FILLED);

    int fillW = static_cast<int>(val * CTRL_W);
    if (fillW > 0)
        cv::rectangle(m_canvas, { CTRL_X, ty, fillW, 6 }, SLIDER_FILL, cv::FILLED);

    // Handle
    int hx = CTRL_X + fillW;
    int hy = ty + 3;
    bool active = (m_activeSlider == fi);
    cv::circle(m_canvas, { hx, hy }, active ? 9 : 7,
        active ? ACCENT : TEXT_PRIMARY, cv::FILLED);
    cv::circle(m_canvas, { hx, hy }, active ? 9 : 7, BORDER, 1);

    // Lưu vùng track để mouse handling
	cv::Rect hitbox = { CTRL_X, yTop, CTRL_W, 32 };
    m_sliderAreas.push_back({ hitbox, fi });
}

// Progress bar 
void CBIRGui::drawProgressBar(int x, int y, int w, int h,
    float val, const cv::Scalar& fill) {
    cv::rectangle(m_canvas, { x, y, w, h }, SLIDER_BG, cv::FILLED);
    int fw = static_cast<int>(max(0.f, min(1.f, val)) * w);
    if (fw > 0) cv::rectangle(m_canvas, { x, y, fw, h }, fill, cv::FILLED);
}

// ============================================================================
//  Panel phải
// ============================================================================
void CBIRGui::drawResultArea() {
    drawQueryPreview();

    int gridX = PANEL_W + 10;
    int gridY = 258;
    int gridW = RESULT_W - 20;
    int gridH = WIN_H - 290;
    drawResultGrid(gridX, gridY, gridW, gridH);
}

// Query preview + weight bars
void CBIRGui::drawQueryPreview() {
    int px = PANEL_W + 12, py = 10;
    int pw = 190, ph = 190;

    // Khung
    cv::rectangle(m_canvas, { px - 2, py - 2, pw + 4, ph + 4 }, BORDER, 1);

    if (!m_queryImg.empty()) {
        cv::Mat thumb;
        cv::resize(m_queryImg, thumb, { pw, ph });
        thumb.copyTo(m_canvas(cv::Rect(px, py, pw, ph)));
    }
    else {
        cv::rectangle(m_canvas, { px, py, pw, ph }, { 42, 42, 52 }, cv::FILLED);
        textCentered(m_canvas, "No Image", { px, py, pw, ph }, 0.50, TEXT_DIM);
    }

    // Label
    cv::putText(m_canvas, "Query Image",
        { px, py + ph + 18 }, cv::FONT_HERSHEY_SIMPLEX,
        0.45, ACCENT, 1, cv::LINE_AA);

    // Thời gian
    if (m_queryTimeMs > 0.0)
        cv::putText(m_canvas, "Time: " + fmtTime(m_queryTimeMs),
            { px, py + ph + 34 }, cv::FONT_HERSHEY_SIMPLEX,
            0.42, TEXT_PRIMARY, 1, cv::LINE_AA);

    // Số kết quả
    if (!m_results.empty())
        cv::putText(m_canvas,
            "Results: " + to_string(m_results.size()),
            { px, py + ph + 52 }, cv::FONT_HERSHEY_SIMPLEX,
            0.34, TEXT_DIM, 1, cv::LINE_AA);
}

// Result grid 
void CBIRGui::drawResultGrid(int gx, int gy, int gw, int gh) {
    // Divider
    cv::line(m_canvas, { PANEL_W, gy - 4 }, { WIN_W, gy - 4 }, BORDER, 1);
    cv::putText(m_canvas, "Search Results  (sorted by similarity score)",
        { gx, gy - 8 }, cv::FONT_HERSHEY_SIMPLEX,
        0.45, TEXT_DIM, 1, cv::LINE_AA);

    if (m_results.empty()) {
        cv::Rect area = { gx, gy, gw, gh };
        cv::rectangle(m_canvas, area, { 34, 34, 42 }, cv::FILLED);
        textCentered(m_canvas,
            m_queryPath.empty()
            ? "Select an image and press SEARCH"
            : "No results found",
            area, 0.52, TEXT_DIM);
        return;
    }

    int cellW = THUMB_SIZE + 14;
    int cellH = THUMB_SIZE + 52;
    int numCols = max(1, gw / cellW);
    int visRows = gh / cellH;

    int totalRows = ((int)m_results.size() + numCols - 1) / numCols;
    m_scrollRow = max(0, min(m_scrollRow, totalRows - visRows));

    int startIdx = m_scrollRow * numCols;
    int endIdx = min((int)m_results.size(),
        startIdx + (visRows + 1) * numCols);

    for (int i = startIdx; i < endIdx; ++i) {
        int local = i - startIdx;
        int col = local % numCols;
        int row = local / numCols;
        int tx = gx + col * cellW;
        int ty = gy + row * cellH;
        if (ty + cellH > gy + gh) break;

        cv::Mat img = cv::imread(m_results[i].imagePath);
        drawThumb(img, tx, ty, m_results[i].score,
            i + 1, m_hoveredThumb == i,
            m_results[i].imagePath);
    }

    // Scroll bar dọc
    if (totalRows > visRows) {
        int sx = gx + gw + 2;
        int sy = gy, sh = gh;
        cv::rectangle(m_canvas, { sx, sy, 8, sh }, SLIDER_BG, cv::FILLED);
        float ratio = (float)m_scrollRow / max(1, totalRows - visRows);
        int   tH = max(20, sh * visRows / totalRows);
        int   tY = sy + (int)((sh - tH) * ratio);
        cv::rectangle(m_canvas, { sx, tY, 8, tH }, SLIDER_FILL, cv::FILLED);
    }
}

// Thumbnail 
void CBIRGui::drawThumb(const cv::Mat& img, int x, int y,
    float score, int rank, bool hovered,
    const string& path) {
    int tw = THUMB_SIZE, th = THUMB_SIZE;
    cv::Scalar cellBg = hovered ? cv::Scalar{ 52,52,68 } : cv::Scalar{ 38,38,48 };
    cv::rectangle(m_canvas, { x - 2, y - 2, tw + 4, th + 54 }, cellBg, cv::FILLED);

    cv::Scalar bdr = hovered ? ACCENT : scoreColor(score);
    int bdrW = hovered ? 2 : 1;
    cv::rectangle(m_canvas, { x - 2, y - 2, tw + 4, th + 54 }, bdr, bdrW);

    // Ảnh
    if (!img.empty()) {
        cv::Mat thumb;
        cv::resize(img, thumb, { tw, th }, 0, 0, cv::INTER_CUBIC);
        thumb.copyTo(m_canvas(cv::Rect(x, y, tw, th)));
    }
    else {
        cv::rectangle(m_canvas, { x, y, tw, th }, { 48,48,58 }, cv::FILLED);
        textCentered(m_canvas, "N/A", { x,y,tw,th }, 0.45, TEXT_DIM);
    }

    // Rank badge
    string rStr = "#" + std::to_string(rank);
    cv::rectangle(m_canvas, { x, y, 28, 17 }, { 18,18,22,180 }, cv::FILLED);
    cv::putText(m_canvas, rStr, { x + 3, y + 13 },
        cv::FONT_HERSHEY_SIMPLEX, 0.38, ACCENT, 1, cv::LINE_AA);

    // Score text
    string sStr = "Score: " + fmtFloat(score, 3);
    cv::putText(m_canvas, sStr, { x, y + th + 14 },
        cv::FONT_HERSHEY_SIMPLEX, 0.40, TEXT_PRIMARY, 1, cv::LINE_AA);

    // Mini score bar
    drawProgressBar(x, y + th + 20, tw, 6, max(0.f, min(1.f, score)),
        scoreColor(score));

    // Filename
    std::string fname = path;
    size_t pos = fname.find_last_of("/\\");
    if (pos != string::npos) fname = fname.substr(pos + 1);
    if (fname.size() > 22) fname = fname.substr(0, 20) + "..";
    cv::putText(m_canvas, fname, { x, y + th + 38 },
        cv::FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv::LINE_AA);
}

// Status bar
void CBIRGui::drawStatusBar() {
    int sy = WIN_H - 26;
    cv::rectangle(m_canvas, { 0, sy, WIN_W, 26 }, { 20,20,26 }, cv::FILLED);
    cv::line(m_canvas, { 0,sy }, { WIN_W,sy }, BORDER, 1);

    string status;
    if (m_queryPath.empty())
        status = "Ready -- Click 'Browse File...' to select a query image.";
    else if (m_results.empty())
        status = "Image loaded -- Press SEARCH or Enter to find similar images.";
    else {
        status = "Top-" + std::to_string(m_results.size())
            + " results   |   Query time: " + fmtTime(m_queryTimeMs);
        if (m_lastAP >= 0.0)
            status += "   |   AP@" + to_string(m_K) + ": " + fmtFloat((float)m_lastAP, 3);
        if (m_mapScore >= 0.0)
            status += "   |   MAP@" + to_string(m_K) + ": " + fmtFloat((float)m_mapScore, 3)
                    + " (" + to_string(m_mapCount) + " queries)";
        status += "   |   Scroll to browse   |   ESC to quit";
    }

    cv::putText(m_canvas, status,
        { PANEL_W + 8, sy + 17 }, cv::FONT_HERSHEY_SIMPLEX,
        0.40, TEXT_DIM, 1, cv::LINE_AA);
    cv::putText(m_canvas, "CSC16105 CBIR v1.0",
        { 8, sy + 17 }, cv::FONT_HERSHEY_SIMPLEX,
        0.38, TEXT_DIM, 1, cv::LINE_AA);
}

// Overlay message
void CBIRGui::showOverlayMsg(const string& msg) {
    cv::Rect box = { PANEL_W + 20, WIN_H / 2 - 22, RESULT_W - 40, 44 };
    cv::rectangle(m_canvas, box, { 32,32,40 }, cv::FILLED);
    cv::rectangle(m_canvas, box, BORDER, 1);
    textCentered(m_canvas, msg, box, 0.52, ACCENT);
    cv::imshow(m_windowTitle, m_canvas);
    cv::waitKey(1);
}

// ============================================================================
//  Mouse handling
//============================================================================
void CBIRGui::onMouse(int event, int x, int y, int flags, void* ud) {
    static_cast<CBIRGui*>(ud)->handleMouse(event, x, y, flags);
}

void CBIRGui::handleMouse(int event, int x, int y, int flags) {
    bool needRedraw = false;

    // Scroll wheel
    if (event == cv::EVENT_MOUSEWHEEL) {
        if (x > PANEL_W) {
            m_scrollRow += (cv::getMouseWheelDelta(flags) > 0) ? -1 : 1;
            m_scrollRow = max(0, m_scrollRow);
            needRedraw = true;
        }
        goto redraw;
    }

    // Move: hover + slider drag 
    if (event == cv::EVENT_MOUSEMOVE) {
        // Hover thumbnail
        {
            int prev = m_hoveredThumb;
            m_hoveredThumb = -1;
            if (x > PANEL_W && !m_results.empty()) {
                int gx = PANEL_W + 10, gy = 258, gw = RESULT_W - 20;
                int cellW = THUMB_SIZE + 14, cellH = THUMB_SIZE + 52;
                int numCols = max(1, gw / cellW);
                int col = (x - gx) / cellW;
                int row = (y - gy) / cellH;
                int idx = (m_scrollRow + row) * numCols + col;
                if (col >= 0 && col < numCols && y > gy &&
                    idx >= 0 && idx < (int)m_results.size())
                    m_hoveredThumb = idx;
            }
            if (m_hoveredThumb != prev) needRedraw = true;
        }
        // Slider drag
        if (m_dragging && m_activeSlider >= 0) {
            float nv = (float)(x - CTRL_X) / (float)CTRL_W;
            m_weights[m_activeSlider] = clampW(nv);
            needRedraw = true;
        }
        goto redraw;
    }

    // Mouse down
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Slider
        for (auto& sl : m_sliderAreas) {
            if (hit(sl.track, x, y)) {
                m_activeSlider = sl.featureIdx;
                m_dragging = true;
                float nv = (float)(x - CTRL_X) / (float)CTRL_W;
                m_weights[sl.featureIdx] = clampW(nv);
                needRedraw = true;
                goto redraw;
            }
        }
        // Browse
        if (hit(m_browseRect, x, y)) {
            string p = openFileDialog();
            if (!p.empty()) {
                m_queryPath = p;
                m_queryImg = cv::imread(p);
                m_results.clear();
                m_queryTimeMs = 0.0;
                m_scrollRow = 0;
            }
            needRedraw = true; goto redraw;
        }
        // Search
        if (hit(m_searchRect, x, y)) { doSearch(); return; }
        // Reset
        if (hit(m_resetRect, x, y)) {
            m_weights = { 0.08f, 0.14f, 0.14f, 0.08f, 0.14f, 0.14f, 0.28f };
            needRedraw = true; goto redraw;
        }
        // Normalize (nút normalize nằm bên phải nút reset, cùng y, cùng height)
        {
            cv::Rect normRect = { CTRL_X + (CTRL_W - 6) / 2 + 6, m_resetRect.y,
                                 (CTRL_W - 6) / 2, 26 };
            if (hit(normRect, x, y)) {
                normalizeWeights();
                needRedraw = true; goto redraw;
            }
        }
        // K−
        if (hit(m_kMinusRect, x, y)) {
            m_K = max(1, m_K - 1);
            needRedraw = true; goto redraw;
        }
        // K+
        if (hit(m_kPlusRect, x, y)) {
            m_K = min(50, m_K + 1);
            needRedraw = true; goto redraw;
        }
        // Reset MAP
        if (hit(m_resetMapRect, x, y)) {
            m_lastAP   = -1.0;
            m_mapScore = -1.0;
            m_mapCount = 0;
            needRedraw = true; goto redraw;
        }
    }

    // Mouse up
    if (event == cv::EVENT_LBUTTONUP) {
        m_dragging = false;
        m_activeSlider = -1;
        needRedraw = true;
    }

redraw:
    if (needRedraw) {
        render();
        cv::imshow(m_windowTitle, m_canvas);
    }
}

// ============================================================================
//  Search
// ============================================================================
void CBIRGui::doSearch() {
    if (m_queryPath.empty() || m_queryImg.empty()) {
        showOverlayMsg("Please select a query image first.");
        cv::waitKey(700); return;
    }
    if (!m_searchCb) {
        showOverlayMsg("Search engine not connected.");
        cv::waitKey(700); return;
    }

    showOverlayMsg("Searching ...  please wait");

    normalizeWeights();
    vector<float> wv(m_weights.begin(), m_weights.end());
    InferenceOutput out = m_searchCb(m_queryPath, wv, m_K);

    m_results      = out.results;
    m_queryTimeMs  = out.executionTimeMs;
    m_scrollRow    = 0;

    // Cập nhật AP và MAP tích lũy
    if (out.apScore >= 0.0) {
        m_lastAP = out.apScore;
        // MAP = trung bình trượt (cộng dồn)
        if (m_mapCount == 0) {
            m_mapScore = m_lastAP;
        } else {
            m_mapScore = (m_mapScore * m_mapCount + m_lastAP) / (m_mapCount + 1);
        }
        m_mapCount++;
    }

    render();
    cv::imshow(m_windowTitle, m_canvas);
}

// Normalize weights
void CBIRGui::normalizeWeights() {
    float total = 0.f;
    for (float w : m_weights) total += w;
    if (total < 1e-6f) {
        for (float& w : m_weights) w = 1.f / 7.f;
        return;
    }
    for (float& w : m_weights) w /= total;
}

// ============================================================================
//  MAP Panel — góc trên phải, cạnh Query Preview
// ============================================================================
void CBIRGui::drawMAPPanel() {
    // Vị trí: bên phải query preview (px=PANEL_W+12, pw=190)
    // Đặt panel bắt đầu từ x = PANEL_W + 12 + 190 + 16
    int px = PANEL_W + 220;
    int py = 10;
    int pw = WIN_W - px - 12;   // co giãn đến cạnh phải
    int ph = 230;               // cùng chiều cao vùng preview

    // Nền panel
    cv::rectangle(m_canvas, { px, py, pw, ph }, { 32, 32, 42 }, cv::FILLED);
    cv::rectangle(m_canvas, { px, py, pw, ph }, BORDER, 1);

    // Thanh màu nhấn trái
    cv::rectangle(m_canvas, { px, py, 3, ph }, ACCENT, cv::FILLED);

    int y = py + 14;

    // Tiêu đề
    cv::putText(m_canvas, "Evaluation Metrics",
        { px + 10, y }, cv::FONT_HERSHEY_SIMPLEX,
        0.50, ACCENT, 1, cv::LINE_AA);
    y += 6;
    cv::line(m_canvas, { px + 10, y }, { px + pw - 10, y }, BORDER, 1);
    y += 16;

    // ── AP@K ──────────────────────────────────────────────────────────────
    cv::putText(m_canvas, "AP@K  (last query)",
        { px + 10, y }, cv::FONT_HERSHEY_SIMPLEX,
        0.40, TEXT_DIM, 1, cv::LINE_AA);
    y += 18;

    if (m_lastAP >= 0.0) {
        // Số AP
        string apStr = fmtFloat((float)m_lastAP, 4);
        cv::putText(m_canvas, apStr,
            { px + 10, y }, cv::FONT_HERSHEY_SIMPLEX,
            0.70, scoreColor((float)m_lastAP), 2, cv::LINE_AA);
        y += 8;

        // Progress bar AP
        int barW = pw - 20;
        drawProgressBar(px + 10, y, barW, 10,
            (float)m_lastAP, scoreColor((float)m_lastAP));
        y += 20;
    } else {
        cv::putText(m_canvas, "N/A",
            { px + 10, y }, cv::FONT_HERSHEY_SIMPLEX,
            0.60, TEXT_DIM, 1, cv::LINE_AA);
        y += 28;
    }

    y += 8;
    cv::line(m_canvas, { px + 10, y }, { px + pw - 10, y }, BORDER, 1);
    y += 14;

    // ── MAP@K ─────────────────────────────────────────────────────────────
    cv::putText(m_canvas, "MAP@K  (session avg)",
        { px + 10, y }, cv::FONT_HERSHEY_SIMPLEX,
        0.40, TEXT_DIM, 1, cv::LINE_AA);
    y += 18;

    if (m_mapScore >= 0.0) {
        string mapStr = fmtFloat((float)m_mapScore, 4);
        cv::putText(m_canvas, mapStr,
            { px + 10, y }, cv::FONT_HERSHEY_SIMPLEX,
            0.70, scoreColor((float)m_mapScore), 2, cv::LINE_AA);
        y += 8;

        int barW = pw - 20;
        drawProgressBar(px + 10, y, barW, 10,
            (float)m_mapScore, scoreColor((float)m_mapScore));
        y += 20;

        // Số lần query
        string cntStr = "(" + to_string(m_mapCount) + " quer" +
                         (m_mapCount == 1 ? "y" : "ies") + ")";
        cv::putText(m_canvas, cntStr,
            { px + 10, y }, cv::FONT_HERSHEY_SIMPLEX,
            0.38, TEXT_DIM, 1, cv::LINE_AA);
        y += 18;
    } else {
        cv::putText(m_canvas, "N/A",
            { px + 10, y }, cv::FONT_HERSHEY_SIMPLEX,
            0.60, TEXT_DIM, 1, cv::LINE_AA);
        y += 28;
    }

    y += 6;
    cv::line(m_canvas, { px + 10, y }, { px + pw - 10, y }, BORDER, 1);
    y += 14;

    // Ghi chú K hiện tại
    cv::putText(m_canvas, "K = " + to_string(m_K),
        { px + 10, y }, cv::FONT_HERSHEY_SIMPLEX,
        0.38, TEXT_DIM, 1, cv::LINE_AA);
    // Reset MAP
    int btnX = px + pw - 76, btnY = y - 14;
    cv::Rect resetMapBtn = { btnX, btnY, 66, 20 };
    drawButton(resetMapBtn, "Reset MAP");
    // Lưu vị trí nút để xử lý click
    // (xử lý trong handleMouse)
    m_resetMapRect = resetMapBtn;
}

// File dialog 
string CBIRGui::openFileDialog() {
    string path;

#if defined(_WIN32) || defined(_WIN64)
    FILE* pipe = _popen(
        "powershell -NoProfile -Command \""
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$dlg = New-Object System.Windows.Forms.OpenFileDialog;"
        "$dlg.Title = 'Select Query Image';"
        "$dlg.Filter = 'Images|*.jpg;*.jpeg;*.png;*.bmp;*.tiff';"
        "if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
        "{ Write-Output $dlg.FileName }\"",
        "r");
    if (pipe) {
        char buf[1024] = {};
        if (fgets(buf, sizeof(buf), pipe)) {
            path = buf;
            while (!path.empty() && (path.back() == '\n' || path.back() == '\r'))
                path.pop_back();
        }
        _pclose(pipe);
    }
#elif defined(__APPLE__)
    FILE* pipe = popen(
        "osascript -e 'tell app \"Finder\" to POSIX path of"
        " (choose file of type {\"public.image\"} with prompt \"Select query image\")'",
        "r");
    if (pipe) {
        char buf[1024] = {};
        if (fgets(buf, sizeof(buf), pipe)) {
            path = buf;
            while (!path.empty() && (path.back() == '\n' || path.back() == '\r'))
                path.pop_back();
        }
        pclose(pipe);
    }
#else
    // Linux: thử zenity => fallback console
    FILE* pipe = popen(
        "zenity --file-selection --title='Select Query Image'"
        " --file-filter='Images (jpg png bmp)|*.jpg *.jpeg *.png *.bmp' 2>/dev/null",
        "r");
    bool gotPath = false;
    if (pipe) {
        char buf[1024] = {};
        if (fgets(buf, sizeof(buf), pipe)) {
            path = buf;
            while (!path.empty() && (path.back() == '\n' || path.back() == '\r'))
                path.pop_back();
            gotPath = !path.empty();
        }
        if (pclose(pipe) != 0) { path.clear(); gotPath = false; }
    }
    if (!gotPath) {
        // Fallback: nhập từ console (cửa sổ OpenCV vẫn hiển thị)
        cout << "\n[Browse] Enter image path: " << flush;
        getline(cin, path);
    }
#endif
    return path;
}