"""
Camera AI nâng cao cho EZVIZ C6N (RTSP) - chạy trên Termux/Debian (proot-distro)
Pipeline: Motion filter -> MobileNet-SSD (person/motorbike) -> LBPH face recognition
          -> heuristic "dắt xe" -> Telegram alert (có cooldown chống spam)

CHUẨN BỊ TRƯỚC KHI CHẠY:
    1. Copy config.example.py -> config.py và điền RTSP_URL, TELEGRAM_TOKEN, CHAT_ID thật vào đó.
       (config.py KHÔNG được đưa lên GitHub - đã có trong .gitignore)
    2. Tải model: git clone https://github.com/chuanqi305/MobileNet-SSD.git models/mobilenet-ssd
    3. Tạo thư mục known_faces/<ten_nguoi>/anh1.jpg, anh2.jpg, ...

Cài thư viện: pip install opencv-contrib-python numpy requests
(trên Termux/Android dùng proot-distro Debian để tránh lỗi build cv2)
"""

import cv2
import time
import os
import threading
import requests
import numpy as np

try:
    from config import RTSP_URL, TELEGRAM_TOKEN, CHAT_ID
except ImportError:
    raise SystemExit(
        "Không tìm thấy config.py. Hãy copy config.example.py thành config.py "
        "và điền thông tin thật vào đó trước khi chạy."
    )

# ================== CẤU HÌNH KHÔNG NHẠY CẢM ==================
MODEL_PROTO = "models/mobilenet-ssd/deploy.prototxt"
MODEL_WEIGHTS = "models/mobilenet-ssd/MobileNetSSD_deploy.caffemodel"
KNOWN_FACES_DIR = "known_faces"

MOTION_THRESHOLD = 4000       # độ nhạy phát hiện chuyển động, tự chỉnh
DETECT_CONFIDENCE = 0.5       # ngưỡng tin cậy của MobileNet-SSD
COOLDOWN_SECONDS = 20         # tránh gửi Telegram liên tục cho cùng 1 sự kiện
RESIZE_WIDTH = 480            # resize frame trước khi chạy AI cho nhanh

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]

# ================== LUỒNG ĐỌC RTSP RIÊNG (chống lag/buffer) ==================
class RTSPStream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(1)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


# ================== TELEGRAM ==================
def send_telegram(image, caption):
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "caption": caption},
                      files={"photo": ("alert.jpg", buf.tobytes())}, timeout=15)
    except requests.RequestException as e:
        print("Lỗi gửi Telegram:", e)


# ================== NẠP MODEL OBJECT DETECTION ==================
def load_detector():
    if not (os.path.exists(MODEL_PROTO) and os.path.exists(MODEL_WEIGHTS)):
        raise FileNotFoundError(
            "Chưa có model MobileNet-SSD. Clone: "
            "git clone https://github.com/chuanqi305/MobileNet-SSD.git models/mobilenet-ssd"
        )
    return cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)


def detect_objects(net, frame):
    """Trả về list (label, confidence, box) cho person/motorbike."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                  0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    results = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < DETECT_CONFIDENCE:
            continue
        class_id = int(detections[0, 0, i, 1])
        label = VOC_CLASSES[class_id]
        if label not in ("person", "motorbike"):
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        results.append((label, float(conf), box.astype(int)))
    return results


# ================== NẠP NHẬN DIỆN KHUÔN MẶT (LBPH) ==================
def load_face_recognizer():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, labels, label_map = [], [], {}
    if not os.path.isdir(KNOWN_FACES_DIR):
        print("Chưa có thư mục known_faces/ -> bỏ qua nhận diện người quen.")
        return face_cascade, None, {}

    for idx, person in enumerate(sorted(os.listdir(KNOWN_FACES_DIR))):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        label_map[idx] = person
        for fname in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            detected = face_cascade.detectMultiScale(img, 1.1, 5)
            for (x, y, w, h) in detected:
                faces.append(img[y:y + h, x:x + w])
                labels.append(idx)

    if faces:
        recognizer.train(faces, np.array(labels))
        print(f"Đã học {len(label_map)} người quen: {list(label_map.values())}")
    else:
        recognizer = None
        print("Không tìm thấy ảnh khuôn mặt hợp lệ trong known_faces/.")

    return face_cascade, recognizer, label_map


def recognize_face(gray_frame, box, face_cascade, recognizer, label_map):
    x1, y1, x2, y2 = box
    person_crop = gray_frame[max(0, y1):y2, max(0, x1):x2]
    if person_crop.size == 0:
        return "Không rõ"
    detected = face_cascade.detectMultiScale(person_crop, 1.1, 5)
    if len(detected) == 0 or recognizer is None:
        return "Không rõ"
    (fx, fy, fw, fh) = detected[0]
    face_img = person_crop[fy:fy + fh, fx:fx + fw]
    try:
        label_id, confidence = recognizer.predict(face_img)
        # LBPH: confidence càng THẤP càng chắc chắn đúng (ngược trực giác)
        if confidence < 70:
            return label_map.get(label_id, "Không rõ")
    except cv2.error:
        pass
    return "Người lạ"


# ================== HEURISTIC PHÁT HIỆN "DẮT XE" ==================
def check_walking_bike(person_boxes, bike_boxes, prev_bike_centers):
    """Người + xe máy overlap, xe di chuyển chậm & đều -> khả năng đang dắt xe."""
    def overlap(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        return max(0, x2 - x1) * max(0, y2 - y1) > 0

    alerts = []
    new_centers = []
    for bike_box in bike_boxes:
        cx = (bike_box[0] + bike_box[2]) // 2
        cy = (bike_box[1] + bike_box[3]) // 2
        new_centers.append((cx, cy))
        for person_box in person_boxes:
            if overlap(person_box, bike_box):
                if prev_bike_centers:
                    nearest = min(prev_bike_centers,
                                  key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
                    speed = ((nearest[0]-cx)**2 + (nearest[1]-cy)**2) ** 0.5
                    if 0 < speed < 15:
                        alerts.append("Nghi vấn: người đang DẮT xe máy")
    return alerts, new_centers


# ================== VÒNG LẶP CHÍNH ==================
def main():
    print("Đang nạp model...")
    net = load_detector()
    face_cascade, recognizer, label_map = load_face_recognizer()
    stream = RTSPStream(RTSP_URL)

    prev_gray = None
    prev_bike_centers = []
    last_alert_time = {"stranger": 0, "motion": 0, "bike": 0}

    print("Bắt đầu giám sát...")
    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.5)
            continue

        frame = cv2.resize(frame, (RESIZE_WIDTH,
                                    int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1])))
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(prev_gray, gray)
        motion_score = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1])
        prev_gray = gray

        if motion_score <= MOTION_THRESHOLD:
            time.sleep(0.2)
            continue

        objects = detect_objects(net, frame)
        person_boxes = [box for label, conf, box in objects if label == "person"]
        bike_boxes = [box for label, conf, box in objects if label == "motorbike"]

        now = time.time()

        for box in person_boxes:
            identity = recognize_face(gray, box, face_cascade, recognizer, label_map)
            if identity in ("Người lạ", "Không rõ") and now - last_alert_time["stranger"] > COOLDOWN_SECONDS:
                send_telegram(frame, f"🚨 Phát hiện người lạ trong khung hình! ({identity})")
                last_alert_time["stranger"] = now
            elif identity not in ("Người lạ", "Không rõ"):
                print(f"Người quen xuất hiện: {identity}")

        bike_alerts, prev_bike_centers = check_walking_bike(person_boxes, bike_boxes, prev_bike_centers)
        if bike_alerts and now - last_alert_time["bike"] > COOLDOWN_SECONDS:
            send_telegram(frame, "🏍️ " + bike_alerts[0])
            last_alert_time["bike"] = now

        if not person_boxes and not bike_boxes and now - last_alert_time["motion"] > COOLDOWN_SECONDS:
            send_telegram(frame, "⚠️ Phát hiện chuyển động (chưa rõ đối tượng)")
            last_alert_time["motion"] = now

        time.sleep(0.3)


if __name__ == "__main__":
    main()
