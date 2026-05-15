#include <opencv2/opencv.hpp>
#include "net.h"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// 1. 객체 정보 구조체 (거리 변수 추가)
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    double distance;
};

// 2. 클래스 이름 (COCO 데이터셋 기준)
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

// --- NMS(중복 제거) 관련 함수 ---
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
// -----------------------------

int main() {
    // [설정 1] 카메라 보정 데이터 (업로드하신 npz 파일 기반)
    // mtx: [[650.12, 0, 320.5], [0, 650.78, 240.2], [0, 0, 1]] 기준 예시
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 650.12, 0, 320.5, 0, 650.78, 240.2, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.1, -0.2, 0, 0, 0.1);

    // [설정 2] 거리 계산 상수
    const double FOCAL_LENGTH = 650.0;     // cameraMatrix[0][0] 값
    const double REAL_PERSON_HEIGHT = 1.7; // 사람의 실제 키(m)

    ncnn::Net yolo;
    // 경로에 한글이 있다면 절대경로 혹은 슬래시(/) 사용 주의
    if (yolo.load_param("yolo26n_ncnn_model/model.ncnn.param") ||
        yolo.load_model("yolo26n_ncnn_model/model.ncnn.bin")) {
        cerr << "모델 로드 실패!" << endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    const int INPUT_SIZE = 320;
    cv::Mat frame, undistorted;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // A. 왜곡 보정 실행
        cv::undistort(frame, undistorted, cameraMatrix, distCoeffs);

        // B. NCNN 전처리
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(undistorted.data,
            ncnn::Mat::PIXEL_BGR2RGB, undistorted.cols, undistorted.rows, INPUT_SIZE, INPUT_SIZE);

        const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        in.substract_mean_normalize(0, norm_vals);

        // C. 추론 실행
        ncnn::Extractor ex = yolo.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);

        // D. 후처리 및 거리 계산
        float x_scale = (float)undistorted.cols / INPUT_SIZE;
        float y_scale = (float)undistorted.rows / INPUT_SIZE;

        std::vector<Object> proposals;
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
                float cx = out.row(0)[i] * x_scale;
                float cy = out.row(1)[i] * y_scale;
                float w = out.row(2)[i] * x_scale;
                float h = out.row(3)[i] * y_scale;

                Object obj;
                obj.rect.x = cx - w * 0.5f;
                obj.rect.y = cy - h * 0.5f;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_id;
                obj.prob = max_score;

                // 거리 계산 (사람 기준)
                if (class_id == 0)
                    obj.distance = (REAL_PERSON_HEIGHT * FOCAL_LENGTH) / h;
                else
                    obj.distance = 0; // 다른 물체는 일단 0으로 표시

                proposals.push_back(obj);
            }
        }

        // E. NMS 및 결과 시각화
        if (!proposals.empty()) {
            qsort_descent_inplace(proposals, 0, proposals.size() - 1);
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, 0.45f);

            for (int i = 0; i < (int)picked.size(); i++) {
                const Object& obj = proposals[picked[i]];

                // 박스 및 라벨 출력
                cv::rectangle(undistorted, obj.rect, cv::Scalar(0, 255, 0), 2);

                string label = string(class_names[obj.label]);
                if (obj.distance > 0) {
                    label += " " + cv::format("%.2fm", obj.distance);
                }

                cv::putText(undistorted, label, cv::Point(obj.rect.x, obj.rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("YOLO26n + Calibration + Distance", undistorted);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}
