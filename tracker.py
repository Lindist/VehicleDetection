"""
ระบบติดตามยานพาหนะและตรวจสอบการจอดแช่ (Parking Violation Tracker)
- ตรวจว่ารถอยู่ในขอบเขตพื้นที่ห้ามจอด (ROI Polygon) หรือไม่
- คำนวณการเคลื่อนที่ (Stationary Detection)
- จัดการเวลานับถอยหลัง (Grace Period Timer)
- แยกสถานะระหว่าง รถพยาบาล (Authorized) กับ รถทั่วไป (Unauthorized)
"""

import time
import math
import cv2
import numpy as np

class VehicleState:
    MOVING = "MOVING"           # รถกำลังเคลื่อนที่ผ่าน
    WAITING = "WAITING"         # รถจอดนิ่ง กำลังนับเวลาผ่อนผัน (Grace Period)
    VIOLATION = "VIOLATION"     # รถทั่วไปจอดเกินเวลาที่กำหนด (ทำผิดกฎ)
    AUTHORIZED = "AUTHORIZED"   # รถพยาบาลฉุกเฉิน (ได้รับอนุญาตให้จอด)

class ParkingTracker:
    def __init__(self, roi_polygon, grace_period_sec=15, movement_threshold_px=25, 
                 authorized_classes=None, unauthorized_classes=None):
        self.roi_polygon = np.array(roi_polygon, dtype=np.int32)
        self.grace_period_sec = grace_period_sec
        self.movement_threshold_px = movement_threshold_px
        
        self.authorized_classes = set(authorized_classes or ["ambulance"])
        self.unauthorized_classes = set(unauthorized_classes or ["car", "bus", "truck", "motorcycle", "van"])
        
        # จัดเก็บข้อมูลรถ: { track_id: dict }
        self.vehicles = {}
        self.next_auto_id = 1

    def is_inside_roi(self, point):
        """ตรวจสอบว่าจุด (x, y) อยู่ในพื้นที่ห้ามจอดหรือไม่"""
        return cv2.pointPolygonTest(self.roi_polygon, (float(point[0]), float(point[1])), False) >= 0

    def update(self, detections):
        """
        อัปเดตข้อมูลรถจากผลลัพธ์ของ YOLO
        detections: list of dict [
            { 'bbox': [x1, y1, x2, y2], 'class_name': str, 'conf': float, 'track_id': Optional[int] }
        ]
        """
        now = time.time()
        active_ids = set()
        violations = []

        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name'].lower()
            track_id = det.get('track_id')

            # ใช้จุดสัมผัสพื้น (กลางล่างของ Bounding Box) เป็นตำแหน่งอ้างอิงของรถ
            x1, y1, x2, y2 = bbox
            contact_pt = (int((x1 + x2) / 2), int(y2))
            center_pt = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # เช็คว่ารถอยู่ในพื้นที่ห้ามจอดหรือไม่ (ตรวจจุดสัมผัสพื้น หรือจุดศูนย์กลาง)
            inside = self.is_inside_roi(contact_pt) or self.is_inside_roi(center_pt)
            if not inside:
                continue

            # ถ้าไม่มี track_id จาก YOLO ให้จับคู่ด้วยตำแหน่ง centroid
            if track_id is None:
                track_id = self._match_or_create_id(center_pt, class_name)

            active_ids.add(track_id)
            is_authorized = (class_name in self.authorized_classes)

            # ข้อมูลรถเดิมหรือเพิ่งตรวจพบใหม่
            if track_id not in self.vehicles:
                self.vehicles[track_id] = {
                    'track_id': track_id,
                    'class_name': class_name,
                    'is_authorized': is_authorized,
                    'first_seen': now,
                    'last_seen': now,
                    'last_position': center_pt,
                    'stationary_start': now,
                    'state': VehicleState.AUTHORIZED if is_authorized else VehicleState.WAITING,
                    'bbox': bbox,
                    'dwell_time': 0.0
                }
            else:
                veh = self.vehicles[track_id]
                veh['last_seen'] = now
                veh['bbox'] = bbox
                veh['class_name'] = class_name
                veh['is_authorized'] = is_authorized

                # ตรวจสอบการเคลื่อนที่
                last_pt = veh['last_position']
                dist_moved = math.hypot(center_pt[0] - last_pt[0], center_pt[1] - last_pt[1])

                if dist_moved > self.movement_threshold_px:
                    # รถขยับเกินเกณฑ์ -> ถือว่ากำลังขยับ/วิ่ง
                    veh['stationary_start'] = now
                    veh['last_position'] = center_pt
                    if not is_authorized:
                        veh['state'] = VehicleState.MOVING
                else:
                    # รถจอดนิ่ง
                    stationary_duration = now - veh['stationary_start']
                    veh['dwell_time'] = stationary_duration

                    if is_authorized:
                        veh['state'] = VehicleState.AUTHORIZED
                    else:
                        if stationary_duration >= self.grace_period_sec:
                            veh['state'] = VehicleState.VIOLATION
                            violations.append(veh)
                        else:
                            veh['state'] = VehicleState.WAITING

        # ลบรถที่ไม่ได้อยู่ในเฟรมเกิน 3 วินาทีออก
        expired_ids = [vid for vid, v in self.vehicles.items() if (now - v['last_seen']) > 3.0]
        for vid in expired_ids:
            del self.vehicles[vid]

        return list(self.vehicles.values()), violations

    def _match_or_create_id(self, center_pt, class_name):
        """ค้นหา track_id ที่ใกล้ที่สุด หรือสร้าง ID ใหม่"""
        min_dist = float('inf')
        matched_id = None
        for vid, v in self.vehicles.items():
            dist = math.hypot(center_pt[0] - v['last_position'][0], center_pt[1] - v['last_position'][1])
            if dist < 80 and dist < min_dist:
                min_dist = dist
                matched_id = vid

        if matched_id is not None:
            return matched_id

        new_id = self.next_auto_id
        self.next_auto_id += 1
        return new_id
