"""
เครื่องมือกำหนดพื้นที่ห้ามจอด (ROI Polygon Setup Tool)
- เปิดภาพจากกล้องหรือวิดีโอ
- คลิกเมาส์เพื่อวางจุดขอบเขตพื้นที่ห้ามจอด
- บันทึกพิกัดลงไฟล์ config.yaml โดยอัตโนมัติ
- สามารถเรียกใช้เดี่ยวๆ หรือเรียกใช้จาก main.py ได้

วิธีใช้งาน:
- คลิกซ้าย: วางจุดพิกัด
- คลิกขวา: ลบจุดล่าสุด
- กด 'r': ล้างจุดทั้งหมด
- กด 's': บันทึกลง config.yaml แล้วเสร็จสิ้น
- กด 'q': ยกเลิกการแก้ไข
"""

import cv2
import yaml
import numpy as np
import os

CONFIG_PATH = "config.yaml"

points = []

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def save_config(cfg, new_polygon):
    cfg["roi_polygon"] = new_polygon
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True, sort_keys=False)
    print(f"[OK] Saved ROI Polygon to {CONFIG_PATH}: {new_polygon}")

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"[POINT ADDED] P{len(points)}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            removed = points.pop()
            print(f"[POINT UNDO] Removed: {removed}")

def interactive_zone_setup(cap=None, base_frame=None, existing_points=None, window_name="Zone Setup"):
    """
    ฟังก์ชันเปิดหน้าต่าง Interactive ให้ผู้ใช้คลิกกำหนดจุดพื้นที่ห้ามจอดใหม่
    คืนค่าพิกัด Polygon ใหม่เมื่อกด 's' และบันทึกเข้า config.yaml
    """
    global points
    cfg = load_config()

    if existing_points:
        points = [list(pt) for pt in existing_points]
    else:
        points = [list(pt) for pt in cfg.get("roi_polygon", [])]

    # ถ้าไม่มี cap ให้เปิดใหม่จาก config
    created_own_cap = False
    if cap is None or not cap.isOpened():
        source = cfg.get("camera", {}).get("source", 0)
        cap = cv2.VideoCapture(source)
        created_own_cap = True

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    saved_polygon = None
    print(f"\n[ZONE SETUP] เข้าสู่โหมดกำหนดพื้นที่: คลิกซ้ายเพื่อวางจุด, 's' เพื่อบันทึก, 'r' เพื่อรีเซ็ต, 'q' เพื่อยกเลิก")

    while True:
        if cap is not None and cap.isOpened():
            ret, current_frame = cap.read()
            if ret:
                display_frame = current_frame.copy()
            elif base_frame is not None:
                display_frame = base_frame.copy()
            else:
                display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        elif base_frame is not None:
            display_frame = base_frame.copy()
        else:
            display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # วาดเส้นและพื้นที่แรงเงา
        overlay = display_frame.copy()
        if len(points) > 1:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.fillPoly(overlay, [pts], color=(0, 255, 255))
            cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)

        # วาดจุดและหมายเลขกำกับ
        for idx, pt in enumerate(points):
            cv2.circle(display_frame, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(display_frame, f"P{idx+1}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # แถบคำแนะนำ
        guide_bar = display_frame.copy()
        cv2.rectangle(guide_bar, (0, 0), (display_frame.shape[1], 35), (20, 20, 20), -1)
        cv2.addWeighted(guide_bar, 0.75, display_frame, 0.25, 0, display_frame)

        cv2.putText(display_frame, "L-Click: Add | R-Click: Undo | 's': Save & Apply | 'r': Reset | 'q': Cancel",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            if len(points) >= 3:
                save_config(cfg, points)
                saved_polygon = [list(p) for p in points]
                print(f"[ZONE SETUP] บันทึกและนำพื้นที่ใหม่ไปใช้ทันที: {saved_polygon}")
                break
            else:
                print("[WARN] ต้องกำหนดจุดอย่างน้อย 3 จุดขึ้นไป (เพื่อสร้างรูปหลายเหลี่ยม)")
        elif key == ord('r'):
            points = []
            print("[RESET] ล้างจุดทั้งหมดแล้ว")
        elif key == ord('q'):
            print("[CANCEL] ยกเลิกการตั้งค่าพื้นที่ใหม่")
            break

    cv2.destroyWindow(window_name)
    if created_own_cap and cap is not None:
        cap.release()

    return saved_polygon

def main():
    interactive_zone_setup()

if __name__ == "__main__":
    main()
