# v11: 초해상도(Super-Resolution) 및 모든 기능 통합
# v11: CSRT 추적기 적용, 이미지 스태킹(다중 프레임 평균화) 및 모든 기능 통합

# 1. 차량 검출 (YOLOv3-tiny, CPU): 영상 전체에서 차량의 위치를 찾아냅니다.
# 2. 번호판 위치 검출 (YOLOv5, GPU): 1단계에서 찾은 차량 이미지 안에서, 더 정교한 YOLOv5 모델이 번호판의 정확한 위치를 찾아냅니다.
# 3. 화질 개선 (초해상도 SR, GPU): 2단계에서 찾아낸 저화질의 번호판 이미지를 입력받아, 화질을 선명하게 개선합니다.
# 4. 문자 인식 (EasyOCR, GPU): 3단계에서 선명해진 번호판 이미지에서 글자를 읽어 텍스트로 변환합니다. (예: 64가1511 또는 6471511 등)

import cv2
import numpy as np
import os
from collections import deque, Counter
import re
import torch
import easyocr
import warnings

# --- 경고 메시지 숨기기 ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===================================================================
# ✨ 1. 시스템 설정 및 파라미터 (이곳에서 쉽게 튜닝하세요) ✨
# ===================================================================
CONFIG = {
    # --- 파일 경로 ---
    "video_input": '163614_D.avi',
    "video_output": 'output_v11_163614_D.avi',
    "log_file": "log.txt",
    "yolo_vehicle_weights": "yolov3-tiny.weights",
    "yolo_vehicle_cfg": "yolov3-tiny.cfg",
    "coco_names": "coco.names",
    "yolo_plate_model": "lp_det.pt",
    "sr_model": "EDSR_x4.pb",

    # --- 출력 영상 설정 ---
    "output_width": 640,
    "output_height": 480,

    # --- 탐지 및 추적 파라미터 ---
    "conf_threshold_vehicle": 0.5,
    "nms_threshold_vehicle": 0.4,
    "min_vehicle_w": 70,
    "min_vehicle_h": 70,
    "lane_iou_threshold": 0.05,
    "detection_interval": 15,  # 프레임 단위

    # --- 번호판 인식(LPR) 파라미터 ---
    "min_plate_area_for_ocr": 1200,
    "plate_history_size": 7,
    "plate_confirm_threshold": 3,

    # --- 차선 감지 파라미터 ---
    "lane_history_size": 7
}

# --- 2. 모델 로드 ---
try:
    vehicle_net = cv2.dnn.readNet(CONFIG["yolo_vehicle_weights"], CONFIG["yolo_vehicle_cfg"])
    print("[정보] 차량 검출(YOLOv3-tiny) 모델 로드 완료.")
except Exception as e:
    print(f"[오류] 차량 검출 모델 로드 실패: {e}"); exit()

try:
    plate_detector_model = torch.hub.load('ultralytics/yolov5', 'custom', path=CONFIG["yolo_plate_model"],
                                          force_reload=False)
    plate_detector_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[정보] 번호판 검출(YOLOv5) 모델 로드 완료 (백엔드: {'cuda' if torch.cuda.is_available() else 'cpu'}).")
except Exception as e:
    print(f"[오류] 번호판 검출 모델 로드 실패: {e}"); plate_detector_model = None

try:
    ocr_reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
    print(f"[정보] 번호판 인식(EasyOCR) 리더 로드 완료 (백엔드: {'cuda' if torch.cuda.is_available() else 'cpu'}).")
except Exception as e:
    print(f"[오류] EasyOCR 리더 로드 실패: {e}"); ocr_reader = None

try:
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(CONFIG["sr_model"])
    model_name = os.path.basename(CONFIG["sr_model"]).split('_')[0].lower()
    model_scale = int(re.findall(r'x(\d+)', os.path.basename(CONFIG["sr_model"]))[0])
    sr.setModel(model_name, model_scale)
    print(f"[정보] 초해상도({model_name.upper()} x{model_scale}) 모델 로드 완료.")
except Exception as e:
    print(f"[경고] 초해상도 모델 로드 실패: {e}."); sr = None

classes = [];
with open(CONFIG["coco_names"], "r") as f: classes = [line.strip() for line in f.readlines()]
layer_names = vehicle_net.getLayerNames()
try:
    output_layers_indices = vehicle_net.getUnconnectedOutLayers()
    if isinstance(output_layers_indices, np.ndarray) and output_layers_indices.ndim == 2:
        output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]
    else:
        output_layers = [layer_names[i - 1] for i in output_layers_indices]
except AttributeError:
    output_layers = [layer_names[i[0] - 1] for i in vehicle_net.getUnconnectedOutLayers()]

# --- 보조 함수 정의 ---
left_lane_history = deque(maxlen=CONFIG["lane_history_size"])
right_lane_history = deque(maxlen=CONFIG["lane_history_size"])


def detect_lanes(image_param, left_history, right_history):
    image_for_lanes = image_param.copy();
    height, width = image_for_lanes.shape[:2]
    roi_vertices = [(0, height), (int(width / 2), int(height / 1.8)), (width, height)]
    mask = np.zeros_like(image_for_lanes[:, :, 0]);
    cv2.fillPoly(mask, [np.array(roi_vertices, np.int32)], 255)
    masked_image = cv2.bitwise_and(image_for_lanes, image_for_lanes, mask=mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY);
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150);
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, 30, 100)
    lane_image_overlay = np.zeros_like(image_for_lanes);
    left_fit, right_fit = [], []
    current_left_line, current_right_line = None, None
    if lines is not None:
        for line_segment in lines:
            x1, y1, x2, y2 = line_segment[0];
            if x2 - x1 == 0: continue
            slope = (y2 - y1) / (x2 - x1)
            if -0.8 < slope < -0.4:
                left_fit.append((slope, y1 - slope * x1))
            elif 0.4 < slope < 0.8:
                right_fit.append((slope, y1 - slope * x1))
    if left_fit:
        left_fit_avg = np.average(left_fit, axis=0)
        if left_fit_avg[0] != 0:
            y1_l, y2_l = height, int(height / 1.5);
            x1_l, x2_l = int((y1_l - left_fit_avg[1]) / left_fit_avg[0]), int(
                (y2_l - left_fit_avg[1]) / left_fit_avg[0])
            current_left_line = (x1_l, y1_l, x2_l, y2_l);
            left_history.append(current_left_line)
    if right_fit:
        right_fit_avg = np.average(right_fit, axis=0)
        if right_fit_avg[0] != 0:
            y1_r, y2_r = height, int(height / 1.5);
            x1_r, x2_r = int((y1_r - right_fit_avg[1]) / right_fit_avg[0]), int(
                (y2_r - right_fit_avg[1]) / right_fit_avg[0])
            current_right_line = (x1_r, y1_r, x2_r, y2_r);
            right_history.append(current_right_line)
    left_line_available, right_line_available = False, False
    if left_history:
        smoothed_x1_l, smoothed_y1_l = int(np.mean([l[0] for l in left_history])), int(
            np.mean([l[1] for l in left_history]))
        smoothed_x2_l, smoothed_y2_l = int(np.mean([l[2] for l in left_history])), int(
            np.mean([l[3] for l in left_history]))
        cv2.line(lane_image_overlay, (smoothed_x1_l, smoothed_y1_l), (smoothed_x2_l, smoothed_y2_l), (0, 255, 0), 10);
        left_line_available = True
    elif current_left_line:
        cv2.line(lane_image_overlay, (current_left_line[0], current_left_line[1]),
                 (current_left_line[2], current_left_line[3]), (0, 255, 0), 10); left_line_available = True
    if right_history:
        smoothed_x1_r, smoothed_y1_r = int(np.mean([l[0] for l in right_history])), int(
            np.mean([l[1] for l in right_history]))
        smoothed_x2_r, smoothed_y2_r = int(np.mean([l[2] for l in right_history])), int(
            np.mean([l[3] for l in right_history]))
        cv2.line(lane_image_overlay, (smoothed_x1_r, smoothed_y1_r), (smoothed_x2_r, smoothed_y2_r), (0, 255, 0), 10);
        right_line_available = True
    elif current_right_line:
        cv2.line(lane_image_overlay, (current_right_line[0], current_right_line[1]),
                 (current_right_line[2], current_right_line[3]), (0, 255, 0), 10); right_line_available = True
    lane_area_coords = None
    if left_line_available and right_line_available:
        lane_center_x_dynamic = int((smoothed_x2_l + smoothed_x2_r) / 2) if (smoothed_x2_l and smoothed_x2_r) else int(
            width / 2)
        roi_y_start, roi_y_end = int(height / 1.7), height - 20
        lane_width_estimate = abs(smoothed_x2_r - smoothed_x2_l) if (
                    smoothed_x2_l and smoothed_x2_r and abs(smoothed_x2_r - smoothed_x2_l) > 50) else 200
        lane_area_coords = (max(0, lane_center_x_dynamic - int(lane_width_estimate / 1.8)), roi_y_start,
                            min(width, lane_center_x_dynamic + int(lane_width_estimate / 1.8)), roi_y_end)
    elif left_line_available:
        lane_area_coords = (max(0, smoothed_x2_l - 50), int(height / 1.7), min(width, smoothed_x2_l + 150), height - 20)
    elif right_line_available:
        lane_area_coords = (max(0, smoothed_x2_r - 150), int(height / 1.7), min(width, smoothed_x2_r + 50), height - 20)
    return lane_image_overlay, lane_area_coords

def find_plate_with_yolov5(vehicle_roi, model):
    """YOLOv5 모델을 사용하여 차량 ROI 내에서 번호판을 검출합니다."""
    if model is None or vehicle_roi is None or vehicle_roi.size == 0: return None

    # ✨ 수정: verbose=False 인수 제거 ✨
    results = model(vehicle_roi, size=640)

    best_plate_roi = None;
    highest_conf = 0.0
    if len(results.xyxy[0]) > 0:
        for *box, conf, cls in results.xyxy[0]:
            if conf > highest_conf:
                highest_conf = conf
                x1, y1, x2, y2 = map(int, box)
                best_plate_roi = vehicle_roi[max(0, y1):min(vehicle_roi.shape[0], y2),
                                 max(0, x1):min(vehicle_roi.shape[1], x2)]
    return best_plate_roi


def recognize_plate_with_easyocr(image_param, reader):
    if image_param is None or image_param.size == 0 or reader is None: return ""
    allowlist_chars = "0123456789가나다라마바사아자차카타파하거너더러머버서어저고노도로모보소오조구누두루무부수우주배하허호"
    result = reader.readtext(image_param, detail=0, paragraph=True, allowlist=allowlist_chars)
    if not result: return ""
    plate_text = "".join(result).replace(" ", "")
    cleaned_text = re.sub(r'[^0-9가-힣]', '', plate_text)
    patterns = [re.compile(r'^\d{2,3}[가-힣]{1}\d{4}$'), re.compile(r'^[가-힣]{2}\d{2}[가-힣]{1}\d{4}$')]
    for pattern in patterns:
        if pattern.fullmatch(cleaned_text): return cleaned_text
    return ""


def calculate_iou(box_a_coords, box_b_coords):
    box_a = [box_a_coords[0], box_a_coords[1], box_a_coords[0] + box_a_coords[2], box_a_coords[1] + box_a_coords[3]]
    box_b = [box_b_coords[0], box_b_coords[1], box_b_coords[2], box_b_coords[3]]
    x_a = max(box_a[0], box_b[0]);
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2]);
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - inter_area)
    return inter_area / union_area if union_area > 0 else 0


def get_confirmed_plate_by_char_voting(ocr_history):
    if not ocr_history: return None
    plate_lengths = [len(p) for p in ocr_history]
    if not plate_lengths: return None
    most_common_len = Counter(plate_lengths).most_common(1)[0][0]
    valid_candidates = [p for p in ocr_history if len(p) == most_common_len]
    if not valid_candidates: return None
    confirmed_plate = ""
    for i in range(most_common_len):
        char_at_pos_i = [plate[i] for plate in valid_candidates]
        char_counts = Counter(char_at_pos_i).most_common(1)[0]
        if char_counts[1] >= CONFIG["plate_confirm_threshold"]:
            confirmed_plate += char_counts[0]
        else:
            return None
    patterns = [re.compile(r'^\d{2,3}[가-힣]{1}\d{4}$'), re.compile(r'^[가-힣]{2}\d{2}[가-힣]{1}\d{4}$')]
    for pattern in patterns:
        if pattern.fullmatch(confirmed_plate): return confirmed_plate
    return None


# --- 비디오 설정 및 추적 관련 변수 ---
cap = cv2.VideoCapture(CONFIG["video_input"])
if not cap.isOpened(): print(f"[오류] 비디오 파일({CONFIG['video_input']})을 열 수 없습니다."); exit()
original_fps = cap.get(cv2.CAP_PROP_FPS);
original_fps = 30 if original_fps == 0 else original_fps
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(CONFIG["video_output"], fourcc, original_fps,
                            (CONFIG["output_width"], CONFIG["output_height"]))
play_delay_ms = max(1, int(1000 / original_fps))
DELAY_STEP_MS, MIN_DELAY_MS, MAX_DELAY_MS = 5, 1, 200

tracker = None;
is_tracking = False
tracked_vehicle_info = {'box': None, 'ocr_history': deque(maxlen=CONFIG["plate_history_size"]), 'confirmed_plate': None,
                        'logged': False}
latest_plate_candidate_img = None
FRAME_COUNTER = 0

print(f"\n영상 처리를 시작합니다. (초기 딜레이: {play_delay_ms}ms)");
print("  - ESC: 종료, ↑: 속도 증가, ↓: 속도 감소");
print("-" * 70)

# --- 메인 루프 ---
while True:
    ret, frame_original = cap.read()
    if not ret: print("[정보] 비디오의 끝에 도달했거나 프레임을 읽을 수 없습니다."); break

    FRAME_COUNTER += 1
    processed_frame = frame_original.copy()
    height_frame, width_frame = frame_original.shape[:2]

    lane_overlay_image, current_lane_area = detect_lanes(frame_original, left_lane_history, right_lane_history)
    if lane_overlay_image is not None:
        processed_frame = cv2.addWeighted(processed_frame, 1, lane_overlay_image, 0.7, 0)
    if current_lane_area:
        la_x1, la_y1, la_x2, la_y2 = map(int, current_lane_area)
        cv2.rectangle(processed_frame, (la_x1, la_y1), (la_x2, la_y2), (255, 0, 255), 1)

    # --- 추적 및 탐색 로직 ---
    if is_tracking:
        success, box = tracker.update(frame_original)
        if success:
            tracked_vehicle_info['box'] = tuple(map(int, box))
        else:
            is_tracking = False;
            tracker = None
            tracked_vehicle_info = {'box': None, 'ocr_history': deque(maxlen=CONFIG["plate_history_size"]),
                                    'confirmed_plate': None, 'logged': False}
            print("[정보] 추적 실패. 새로운 타겟을 탐색합니다.")

    if not is_tracking or FRAME_COUNTER % CONFIG["detection_interval"] == 0:
        blob = cv2.dnn.blobFromImage(frame_original, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        vehicle_net.setInput(blob);
        detections = vehicle_net.forward(output_layers)
        object_boxes, object_confidences = [], []
        for out in detections:
            for detection in out:
                scores = detection[5:];
                class_id = np.argmax(scores);
                confidence = scores[class_id]
                if confidence > CONFIG["conf_threshold_vehicle"] and classes[class_id] in ['car', 'truck', 'bus']:
                    center_x, center_y = int(detection[0] * width_frame), int(detection[1] * height_frame)
                    w, h = int(detection[2] * width_frame), int(detection[3] * height_frame)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    object_boxes.append([x, y, w, h]);
                    object_confidences.append(float(confidence))

        all_detected_boxes = []
        if object_boxes:
            indexes = cv2.dnn.NMSBoxes(object_boxes, object_confidences, CONFIG["conf_threshold_vehicle"],
                                       CONFIG["nms_threshold_vehicle"])
            if isinstance(indexes, np.ndarray): all_detected_boxes = [object_boxes[i] for i in indexes.flatten()]

        # 모든 감지된 차량 그리기
        for x, y, w, h in all_detected_boxes:
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        lane_intruding_candidates = []
        if current_lane_area:
            la_x1, la_y1, la_x2, la_y2 = map(int, current_lane_area)
            lane_area_box_for_iou = (la_x1, la_y1, la_x2, la_y2)
            for x, y, w, h in all_detected_boxes:
                if calculate_iou((x, y, w, h), lane_area_box_for_iou) > CONFIG["lane_iou_threshold"] and w > CONFIG[
                    "min_vehicle_w"] and h > CONFIG["min_vehicle_h"]:
                    lane_intruding_candidates.append({'box': (x, y, w, h), 'area': w * h})

        if lane_intruding_candidates:
            lane_intruding_candidates.sort(key=lambda c: c['area'], reverse=True)
            target_to_track_box = lane_intruding_candidates[0]['box']

            # 현재 추적중인 타겟과 새로 찾은 타겟이 다를 경우에만 추적기 리셋
            if not is_tracking or calculate_iou(tracked_vehicle_info['box'], target_to_track_box) < 0.3:
                tracker = cv2.TrackerCSRT_create();
                tracker.init(frame_original, target_to_track_box)
                is_tracking = True
                print("[정보] 새로운 차량 추적 시작/전환!")
                tracked_vehicle_info = {'box': target_to_track_box,
                                        'ocr_history': deque(maxlen=CONFIG["plate_history_size"]),
                                        'confirmed_plate': None, 'logged': False}

    # --- 추적 중인 차량에 대한 처리 ---
    if is_tracking and tracked_vehicle_info['box']:
        x, y, w, h = tracked_vehicle_info['box']
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        if tracked_vehicle_info['confirmed_plate'] is None:
            vehicle_roi = frame_original[max(0, y):min(height_frame, y + h), max(0, x):min(width_frame, x + w)]
            plate_candidate_area = find_plate_with_yolov5(vehicle_roi, plate_detector_model)

            if plate_candidate_area is not None and plate_candidate_area.size > CONFIG["min_plate_area_for_ocr"]:
                if sr is not None:
                    upscaled_plate = sr.upsample(plate_candidate_area)
                else:
                    upscaled_plate = plate_candidate_area
                latest_plate_candidate_img = upscaled_plate.copy()

                plate_text = recognize_plate_with_easyocr(upscaled_plate, ocr_reader)
                if plate_text:
                    tracked_vehicle_info['ocr_history'].append(plate_text)

                if len(tracked_vehicle_info['ocr_history']) >= CONFIG["plate_history_size"]:
                    confirmed_plate = get_confirmed_plate_by_char_voting(list(tracked_vehicle_info['ocr_history']))
                    if confirmed_plate:
                        tracked_vehicle_info['confirmed_plate'] = confirmed_plate
                        print(f"\n✨✨✨ 번호판 확정 (문자 투표): {confirmed_plate} ✨✨✨\n")

    # --- 최종 결과 표시, 저장 및 키 입력 처리 ---
    if 'latest_plate_candidate_img' in locals() and latest_plate_candidate_img is not None:
        h_plate, w_plate = latest_plate_candidate_img.shape[:2]
        if h_plate > 0 and w_plate > 0:
            display_h, display_w = 50, 180
            plate_display = cv2.resize(latest_plate_candidate_img, (display_w, display_h))
            if len(plate_display.shape) == 2: plate_display = cv2.cvtColor(plate_display, cv2.COLOR_GRAY2BGR)
            processed_frame[10:10 + display_h, 10:10 + display_w] = plate_display
            cv2.rectangle(processed_frame, (10, 10), (10 + display_w, 10 + display_h), (0, 255, 255), 2)
    if tracked_vehicle_info['confirmed_plate'] and not tracked_vehicle_info['logged']:
        log_x, log_y, log_w, log_h = tracked_vehicle_info['box']
        log_message = f"번호판: {tracked_vehicle_info['confirmed_plate']} (대상 차량 크기: {log_w}x{log_h})"
        print(log_message);
        with open(CONFIG["log_file"], "a", encoding="utf-8") as log_file: log_file.write(log_message + "\n")
        tracked_vehicle_info['logged'] = True
    if tracked_vehicle_info['confirmed_plate']:
        disp_x, disp_y, _, _ = tracked_vehicle_info['box']
        cv2.putText(processed_frame, tracked_vehicle_info['confirmed_plate'],
                    (disp_x, disp_y - 10 if disp_y > 10 else disp_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255),
                    3)

    output_display_frame = cv2.resize(processed_frame, (CONFIG["output_width"], CONFIG["output_height"]))
    cv2.imshow("Final Output with Tracker", output_display_frame)
    out_video.write(output_display_frame)
    key = cv2.waitKeyEx(play_delay_ms) & 0xFFFFFF
    if key == 27:
        break
    elif key == 0x260000:
        play_delay_ms = max(MIN_DELAY_MS, play_delay_ms - DELAY_STEP_MS); print(f"  딜레이: {play_delay_ms}ms")
    elif key == 0x280000:
        play_delay_ms = min(MAX_DELAY_MS, play_delay_ms + DELAY_STEP_MS); print(f"  딜레이: {play_delay_ms}ms")
    elif key == ord('q'):
        break

# --- 자원 해제 ---
cap.release();
out_video.release();
cv2.destroyAllWindows()
print(f"\n[정보] 처리가 완료되었습니다. 결과 영상은 {CONFIG['video_output']} 에 저장되었습니다.")