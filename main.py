"""
ระบบตรวจจับรถฉุกเฉินและรถธรรมดาที่จอดในพื้นที่ห้ามจอด (Main Pipeline)
- ประมวลผลภาพจากกล้องด้วย YOLOv8
- ตรวจสอบขอบเขตพื้นที่ห้ามจอด (No-Parking Zone)
- ควบคุมเวลานับถอยหลังการจอดแช่
- สั่งเล่นเสียงเตือนผ่านลำโพงทันทีเมื่อพบการฝ่าฝืน
- รองรับปุ่มคลิกบนหน้าจอ และคีย์ลัด 'z' เพื่อเปิดโหมดตั้งค่าขอบเขตพื้นที่ห้ามจอดใหม่ได้ทันที
"""

import os
import sys
import time
import cv2
import yaml
import numpy as np

from audio_player import AudioAlertPlayer
from tracker import ParkingTracker, VehicleState
from zone_setup import interactive_zone_setup

CONFIG_PATH = "config.yaml"

# พิกัดปุ่มบนหน้าจอ [x1, y1, x2, y2]
btn_set_zone = [470, 5, 630, 35]
request_zone_setup = False

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"[ERROR] Config not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def on_main_mouse(event, x, y, flags, param):
    global request_zone_setup
    if event == cv2.EVENT_LBUTTONDOWN:
        # เช็คว่าคลิกโดนปุ่ม [ SET ZONE ] หรือไม่
        if btn_set_zone[0] <= x <= btn_set_zone[2] and btn_set_zone[1] <= y <= btn_set_zone[3]:
            print("\n[CLICK] ผู้ใช้กดปุ่ม 'SET ZONE' บนหน้าจอ!")
            request_zone_setup = True

def main():
    global request_zone_setup
    print("=" * 60)
    print("[SYSTEM] Starting Emergency No-Parking Violation Detection System")
    print("=" * 60)

    cfg = load_config()

    # 1. ตั้งค่าระบบเสียงลำโพง
    audio_cfg = cfg.get("audio", {})
    parking_cfg = cfg.get("parking", {})
    alert_interval = parking_cfg.get("alert_interval_seconds", 8.0)
    audio_player = AudioAlertPlayer(
        audio_file=audio_cfg.get("file", "assets/alert.wav"),
        volume=audio_cfg.get("volume", 1.0),
        min_interval=alert_interval
    )

    # 2. ตั้งค่าตัวติดตามตำแหน่งรถ (Tracker)
    roi_polygon = cfg.get("roi_polygon", [])
    if not roi_polygon or len(roi_polygon) < 3:
        print("[WARN] No ROI polygon defined. Setting default rectangle.")
        roi_polygon = [[100, 100], [540, 100], [540, 400], [100, 400]]

    tracker = ParkingTracker(
        roi_polygon=roi_polygon,
        grace_period_sec=parking_cfg.get("grace_period_seconds", 15),
        movement_threshold_px=parking_cfg.get("movement_threshold_pixels", 25),
        authorized_classes=cfg.get("model", {}).get("authorized_classes", ["ambulance"]),
        unauthorized_classes=cfg.get("model", {}).get("unauthorized_classes", ["car", "bus", "truck", "motorcycle", "van"])
    )

    # 3. โหลดโมเดล YOLO
    from ultralytics import YOLO
    model_cfg = cfg.get("model", {})
    model_path = model_cfg.get("path", "models/best.pt")
    conf_thresh = model_cfg.get("conf", 0.40)

    if not os.path.exists(model_path):
        print(f"[WARN] Model file not found at '{model_path}'")
        print("[INFO] Fallback to standard 'yolov8n.pt' for system testing...")
        model_path = "yolov8n.pt"

    print(f"[AI] Loading YOLO model: {model_path} ...")
    model = YOLO(model_path)
    print("[OK] Model loaded successfully!")

    # 4. เปิดกล้องหรือวิดีโอ
    cam_cfg = cfg.get("camera", {})
    source = cam_cfg.get("source", 0)
    process_fps = cam_cfg.get("process_fps", 3)
    frame_interval = 1.0 / max(1, process_fps)

    print(f"[CAMERA] Connecting to source: {source} (Processing at {process_fps} FPS to save CPU)...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera/video from: {source}")
        sys.exit(1)

    # ตั้งค่าความละเอียดกล้อง (ถ้าเป็นเว็บแคม)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))

    last_process_time = 0.0
    tracked_vehicles = []
    violations = []
    has_violation_active = False

    pts_polygon = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))

    window_title = "Emergency No-Parking Alert System"
    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, on_main_mouse)

    print("\n[READY] System is running!")
    print("  - คลิกปุ่ม [ SET ZONE ] บนจอ หรือกดปุ่ม 'z' เพื่อตั้งค่าจุดพื้นที่ห้ามจอดใหม่")
    print("  - กด 't' เพื่อทดสอบส่งเสียงเตือนลำโพง")
    print("  - กด 'q' เพื่อออกจากโปรแกรม\n")

    while True:
        # ตรวจสอบว่ามีการเรียกโหมดตั้งค่าจุดพื้นที่ห้ามจอดใหม่หรือไม่
        if request_zone_setup:
            request_zone_setup = False
            ret, cur_frame = cap.read()
            new_poly = interactive_zone_setup(
                cap=cap,
                base_frame=cur_frame if ret else None,
                existing_points=roi_polygon,
                window_name="Zone Setup - Setting New Points"
            )
            if new_poly and len(new_poly) >= 3:
                roi_polygon = new_poly
                tracker.roi_polygon = np.array(roi_polygon, dtype=np.int32)
                pts_polygon = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
                tracker.vehicles.clear()  # เคลียร์สถานะรถเก่าเพื่อเริ่มนับกับพื้นที่ใหม่
                print("[OK] อัปเดตพื้นที่ห้ามจอดใหม่ในระบบเรียบร้อยแล้ว!")
            
            # คืนค่าหน้าต่างเดิม
            cv2.namedWindow(window_title)
            cv2.setMouseCallback(window_title, on_main_mouse)
            continue

        ret, frame = cap.read()
        if not ret:
            if not isinstance(source, int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[ERROR] Camera stream interrupted.")
                break

        now = time.time()
        
        # ควบคุมอัตราการรันโมเดล (ประหยัดพลังงานบน Raspberry Pi 4)
        if (now - last_process_time) >= frame_interval:
            last_process_time = now

            try:
                results = model.track(frame, persist=True, conf=conf_thresh, verbose=False)
            except Exception:
                results = model.predict(frame, conf=conf_thresh, verbose=False)

            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        class_name = model.names[cls_id]
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)

                        track_id = None
                        if box.id is not None:
                            track_id = int(box.id[0].item())

                        detections.append({
                            'bbox': xyxy.tolist(),
                            'class_name': class_name,
                            'conf': conf,
                            'track_id': track_id
                        })

            # อัปเดตข้อมูลรถเข้า Tracker
            tracked_vehicles, violations = tracker.update(detections)

            # ตรวจสอบการส่งเสียงเตือน
            has_violation_active = (len(violations) > 0)
            if has_violation_active:
                audio_player.play()

        # ========================================================
        # การวาดส่วนแสดงผล (Visual Overlay / HUD)
        # ========================================================
        display = frame.copy()
        frame_w = display.shape[1]

        # 1. วาดกรอบพื้นที่ห้ามจอด (ROI Polygon)
        has_ambulance = any(v['state'] == VehicleState.AUTHORIZED for v in tracked_vehicles)
        if has_violation_active:
            zone_color = (0, 0, 255)       # แดง (กำลังร้องเตือน)
            zone_text = "ZONE: VIOLATION DETECTED! (ALARM ACTIVE)"
        elif has_ambulance:
            zone_color = (0, 255, 0)       # เขียว (รถพยาบาลใช้งาน)
            zone_text = "ZONE: AMBULANCE AUTHORIZED"
        else:
            zone_color = (255, 200, 0)     # ฟ้า/เหลือง (ปกติ)
            zone_text = "ZONE: EMERGENCY VEHICLES ONLY"

        # วาดเส้นขอบและเฉดสีโปร่งแสง
        overlay = display.copy()
        cv2.fillPoly(overlay, [pts_polygon], zone_color)
        cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
        cv2.polylines(display, [pts_polygon], isClosed=True, color=zone_color, thickness=3)

        # 2. วาด Bounding Box ของรถในพื้นที่
        for v in tracked_vehicles:
            x1, y1, x2, y2 = v['bbox']
            state = v['state']
            cls_name = v['class_name']
            dwell = int(v['dwell_time'])
            grace_max = tracker.grace_period_sec

            if state == VehicleState.AUTHORIZED:
                box_color = (0, 255, 0)
                label = f"AMBULANCE [ALLOWED]"
            elif state == VehicleState.VIOLATION:
                box_color = (0, 0, 255)
                label = f"VIOLATION! {cls_name.upper()} [Parked: {dwell}s]"
            elif state == VehicleState.WAITING:
                box_color = (0, 165, 255)  # สีส้ม
                label = f"{cls_name.upper()} [Grace: {dwell}/{grace_max}s]"
            else:
                box_color = (255, 255, 0)
                label = f"{cls_name.upper()} [Moving]"

            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), box_color, -1)
            cv2.putText(display, label, (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 3. แถบสถานะด้านบน (Top Status HUD)
        hud_bg = display.copy()
        cv2.rectangle(hud_bg, (0, 0), (frame_w, 42), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.75, display, 0.25, 0, display)

        cv2.putText(display, zone_text, (15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, zone_color, 2)

        # ปรับตำแหน่งปุ่มตามขนาดหน้าจอ
        btn_w = 160
        btn_set_zone[0] = frame_w - btn_w - 10
        btn_set_zone[2] = frame_w - 10
        btn_set_zone[1] = 6
        btn_set_zone[3] = 36

        # วาดปุ่ม [ ⚙️ SET ZONE (Z) ] ให้คลิกได้
        cv2.rectangle(display, (btn_set_zone[0], btn_set_zone[1]),
                               (btn_set_zone[2], btn_set_zone[3]), (50, 150, 255), -1)
        cv2.rectangle(display, (btn_set_zone[0], btn_set_zone[1]),
                               (btn_set_zone[2], btn_set_zone[3]), (255, 255, 255), 1)
        cv2.putText(display, "SET ZONE (Z)", (btn_set_zone[0] + 18, btn_set_zone[1] + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        # แสดงหน้าต่าง
        cv2.imshow(window_title, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[EXIT] Stopping system...")
            break
        elif key == ord('t'):
            print("[TEST] Triggering manual alarm sound test...")
            audio_player.play(force=True)
        elif key == ord('z') or key == ord('Z'):
            print("\n[HOTKEY] ผู้ใช้กดปุ่ม 'z' เพื่อเปิดโหมดตั้งค่าพื้นที่ห้ามจอด!")
            request_zone_setup = True

    cap.release()
    cv2.destroyAllWindows()
    audio_player.stop()

if __name__ == "__main__":
    main()
