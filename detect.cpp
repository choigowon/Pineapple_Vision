#include <opencv2/opencv.hpp>
#include "net.h"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// COCO 80 클래스
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

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// NMS 관련 함수들
static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left; int j = right; float p = faceobjects[(left + right) / 2].prob;
    while (i <= j) {
        while (faceobjects[i].prob > p) i++;
        while (faceobjects[j].prob < p) j--;
        if (i <= j) { std::swap(faceobjects[i], faceobjects[j]); i++; j--; }
    }
    if (left < j) qsort_descent_inplace(faceobjects, left, j);
    if (i < right) qsort_descent_inplace(faceobjects, i, right);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = faceobjects[i].rect.area();
    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold) keep = 0;
        }
        if (keep) picked.push_back(i);
    }
}

int main() {
    ncnn::Net yolo;
    // 새 모델 폴더와 파일명에 맞춰 경로 수정
    if (yolo.load_param("yolo26n_ncnn_model/model.ncnn.param") ||
        yolo.load_model("yolo26n_ncnn_model/model.ncnn.bin")) {
        cerr << "YOLO26n 모델 로드 실패!" << endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    const int INPUT_SIZE = 320;
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data,
            ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows, INPUT_SIZE, INPUT_SIZE);

        const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        in.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolo.create_extractor();
        // 만약 에러가 나면 model.ncnn.param 파일을 열어 images/output0 등 실제 이름을 확인하세요
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);

        float x_scale = (float)frame.cols / INPUT_SIZE;
        float y_scale = (float)frame.rows / INPUT_SIZE;

        std::vector<Object> proposals;
        // YOLO26n의 Transpose 데이터 구조 처리
        for (int i = 0; i < out.w; i++) {
            float max_score = 0.f;
            int class_id = 0;
            for (int j = 0; j < 80; j++) {
                float score = out.row(4 + j)[i];
                if (score > max_score) {
                    max_score = score;
                    class_id = j;
                }
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
            qsort_descent_inplace(proposals, 0, proposals.size() - 1);
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, 0.45f);

            for (int i = 0; i < (int)picked.size(); i++) {
                const Object& obj = proposals[picked[i]];
                cv::rectangle(frame, obj.rect, cv::Scalar(255, 0, 0), 2); // YOLO26n은 파란색으로!
                string label = string(class_names[obj.label]) + " " + to_string((int)(obj.prob * 100)) + "%";
                cv::putText(frame, label, cv::Point(obj.rect.x, obj.rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }

        cv::imshow("YOLO26n NCNN Real-time", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}
