#include <opencv2/opencv.hpp>
#include "net.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;

struct Object {
    cv::Rect_<float> rect;
    int label = 0;
    float prob = 0.0f;
    Object() : rect(0, 0, 0, 0), label(0), prob(0.0f) {}
};

static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// NMS 관련 함수들 (안정성 강화)
static inline float intersection_area(const Object& a, const Object& b) noexcept {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) noexcept {
    int i = left; int j = right;
    float p = faceobjects[static_cast<size_t>(left + right) / 2].prob;
    while (i <= j) {
        while (faceobjects[static_cast<size_t>(i)].prob > p) i++;
        while (faceobjects[static_cast<size_t>(j)].prob < p) j--;
        if (i <= j) { std::swap(faceobjects[static_cast<size_t>(i)], faceobjects[static_cast<size_t>(j)]); i++; j--; }
    }
    if (left < j) qsort_descent_inplace(faceobjects, left, j);
    if (i < right) qsort_descent_inplace(faceobjects, i, right);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) noexcept {
    picked.clear();
    const int n = static_cast<int>(faceobjects.size());
    std::vector<float> areas(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) areas[static_cast<size_t>(i)] = faceobjects[static_cast<size_t>(i)].rect.area();
    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[static_cast<size_t>(i)];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[static_cast<size_t>(picked[static_cast<size_t>(j)])];
            float inter_area = intersection_area(a, b);
            float union_area = areas[static_cast<size_t>(i)] + areas[static_cast<size_t>(picked[static_cast<size_t>(j)])] - inter_area;
            if (inter_area / union_area > nms_threshold) keep = 0;
        }
        if (keep) picked.push_back(i);
    }
}

int main() {
    ncnn::Net yolo;
    yolo.opt.num_threads = 4;

    if (yolo.load_param("yolo26n_ncnn_model/model.ncnn.param") ||
        yolo.load_model("yolo26n_ncnn_model/model.ncnn.bin")) {
        return -1;
    }

    cv::VideoCapture cap(0);
    const int INPUT_SIZE = 320;
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = chrono::steady_clock::now();

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data,
            ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows, INPUT_SIZE, INPUT_SIZE);
        const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        in.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolo.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);

        float x_scale = static_cast<float>(frame.cols) / INPUT_SIZE;
        float y_scale = static_cast<float>(frame.rows) / INPUT_SIZE;

        std::vector<Object> proposals;
        for (int i = 0; i < out.w; i++) {
            float max_score = 0.f;
            int class_id = 0;
            for (int j = 0; j < 80; j++) {
                float score = out.row(4 + j)[i];
                if (score > max_score) { max_score = score; class_id = j; }
            }

            if (max_score > 0.45f) {
                Object obj;
                float cx = out.row(0)[i] * x_scale;
                float cy = out.row(1)[i] * y_scale;
                float w = out.row(2)[i] * x_scale;
                float h = out.row(3)[i] * y_scale;
                obj.rect.x = cx - w * 0.5f;
                obj.rect.y = cy - h * 0.5f;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_id;
                obj.prob = max_score;
                proposals.push_back(obj);
            }
        }

        if (!proposals.empty()) {
            qsort_descent_inplace(proposals, 0, static_cast<int>(proposals.size()) - 1);
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, 0.45f);

            for (int i = 0; i < (int)picked.size(); i++) {
                const Object& obj = proposals[static_cast<size_t>(picked[static_cast<size_t>(i)])];

                // 시각화 복구 (사물 이름 표시)
                cv::rectangle(frame, obj.rect, cv::Scalar(255, 0, 0), 2);
                string label = string(class_names[obj.label]) + " " + cv::format("%.1f%%", obj.prob * 100);
                cv::putText(frame, label, cv::Point(obj.rect.x, obj.rect.y - 5), 0, 0.5, cv::Scalar(255, 255, 255), 1);

                // --- [추가] 사물 위치 판별 로직 ---
                float center_x = obj.rect.x + (obj.rect.width / 2);
                string position;
                if (center_x < frame.cols / 3.0) position = "Left";
                else if (center_x < (frame.cols / 3.0) * 2.0) position = "Center";
                else position = "Right";

                // 콘솔에 출력 (예: person is on the Center)
                if (i == 0) { // 가장 정확한 첫 번째 사물만 출력해서 콘솔이 어지럽지 않게 함
                    cout << class_names[obj.label] << " is on the " << position << endl;
                }
            }
        }

        auto end = chrono::steady_clock::now();
        float fps = 1.0f / chrono::duration<float>(end - start).count();
        cv::putText(frame, cv::format("FPS: %.2f", fps), cv::Point(10, 30), 0, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow("YOLO26n Guide System", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}
