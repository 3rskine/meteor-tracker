#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光鬼增強版流星偵測系統 - Light Trail + Kalman + 合併

主要改進:
1. 多幀累積光鬼效果
2. 自適應衰減權重
3. 短軌跡增強偵測
"""
import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict, deque
import copy


# ==================== Kalman 濾波器 ====================
class SimpleKalman:
    """時間感知的 Kalman 濾波器 (常速度模型)"""
    def __init__(self, init_x, init_y, init_vx=0.0, init_vy=0.0, process_var=1.0, meas_var=5.0):
        self.x = np.array([float(init_x), float(init_y), float(init_vx), float(init_vy)], dtype=float)
        self.P = np.eye(4, dtype=float) * 500.0
        self.process_var = float(process_var)
        self.R = np.eye(2, dtype=float) * float(meas_var)

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        q = max(1e-6, self.process_var)
        Q = np.diag([q * dt, q * dt, q, q])
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q
        return self.x.copy()

    def update(self, meas_x, meas_y):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        z = np.array([float(meas_x), float(meas_y)], dtype=float)
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        y = z - H.dot(self.x)
        self.x = self.x + K.dot(y)
        I = np.eye(4, dtype=float)
        self.P = (I - K.dot(H)).dot(self.P)
        return self.x.copy()

    def clone(self):
        new = SimpleKalman(0, 0)
        new.x = self.x.copy()
        new.P = self.P.copy()
        new.process_var = float(self.process_var)
        new.R = self.R.copy()
        return new


# ==================== RANSAC 共線檢查 ====================
def is_collinear_ransac(points, threshold=3.0, min_inliers_ratio=0.6, max_iterations=120):
    """RANSAC 驗證點集是否共線"""
    if len(points) < 2:
        return False
    pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
    if len(pts) <= 2:
        return True

    n = len(pts)
    best_inliers = 0
    for _ in range(max_iterations):
        i, j = np.random.choice(n, 2, replace=False)
        p1 = pts[i]
        p2 = pts[j]
        v = p2 - p1
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-6:
            continue
        distances = np.abs(np.cross(v, pts - p1)) / norm_v
        inliers = np.count_nonzero(distances < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            if best_inliers / n >= min_inliers_ratio:
                return True
    return (best_inliers / n) >= min_inliers_ratio


# ==================== 光鬼處理器 ====================
class LightTrailProcessor:
    """光鬼 (Light Trail) 處理器 - 累積移動物體軌跡"""
    
    def __init__(self, trail_length=5, decay_type='exponential'):
        """
        Args:
            trail_length: 累積幀數 (建議 3-8)
            decay_type: 衰減模式 ('exponential', 'linear', 'uniform')
        """
        self.trail_length = trail_length
        self.decay_type = decay_type
        self.frame_buffer = deque(maxlen=trail_length)
        self.weights = self._compute_weights()
        
    def _compute_weights(self):
        """計算時間衰減權重"""
        n = self.trail_length
        if self.decay_type == 'exponential':
            # 指數衰減：最新幀權重最高
            weights = np.exp(-np.arange(n) * 0.5)
        elif self.decay_type == 'linear':
            # 線性衰減
            weights = np.linspace(0.3, 1.0, n)
        else:  # uniform
            # 均勻權重
            weights = np.ones(n)
        
        # 正規化
        weights = weights / weights.sum()
        return weights[::-1]  # 反轉，最舊的在前
    
    def add_frame(self, frame):
        """加入新幀到緩衝區"""
        self.frame_buffer.append(frame.copy())
    
    def get_light_trail(self):
        """生成光鬼效果影像"""
        if len(self.frame_buffer) == 0:
            return None
        
        # 加權平均（浮點數運算）
        result = np.zeros_like(self.frame_buffer[0], dtype=np.float32)
        
        for i, frame in enumerate(self.frame_buffer):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            result += frame.astype(np.float32) * weight
        
        return result.astype(np.uint8)
    
    def get_max_projection(self):
        """最大值投影 - 保留所有幀中最亮的像素"""
        if len(self.frame_buffer) == 0:
            return None
        
        result = np.zeros_like(self.frame_buffer[0], dtype=np.uint8)
        for frame in self.frame_buffer:
            result = np.maximum(result, frame)
        
        return result
    
    def get_enhanced_trail(self):
        """增強版光鬼：結合加權平均與最大值投影"""
        if len(self.frame_buffer) == 0:
            return None
        
        weighted = self.get_light_trail()
        max_proj = self.get_max_projection()
        
        # 混合兩種效果 (70% 加權 + 30% 最大值)
        enhanced = cv2.addWeighted(weighted, 0.7, max_proj, 0.3, 0)
        
        return enhanced


# ==================== 主要偵測類別（加入光鬼）====================
class MeteorTrackerWithLightTrail:
    def __init__(self, video_path, output_folder, debug=False,
                 light_trail_length=5, light_trail_mode='enhanced'):
        self.video_path = Path(video_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.debug = debug

        # 讀取影片資訊
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片: {video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        cap.release()

        # === 光鬼處理器 ===
        self.light_trail = LightTrailProcessor(
            trail_length=light_trail_length,
            decay_type='exponential'
        )
        self.light_trail_mode = light_trail_mode  # 'weighted', 'max', 'enhanced'

        # 偵測結果
        self.meteors = []
        self.log_file = self.output_folder / "detection.log"
        self.active_tracks = []
        self.meteor_id_counter = 0
        self.scene_change_frames = []

        # === 優化參數（針對短軌跡）===
        self.min_movement_pixels = 2  # 降低 3→2
        self.movement_check_frames = 2  # 降低 6→4，更快判斷
        self.lost_track_threshold = 20  # 提高 15→20，允許更長中斷

        # Multi-frame 差分設定
        self.target_window_seconds = 0.6  # 略短的窗口
        self.gray_buffer = deque(maxlen=max(120, int(self.fps * 4)))

        # 近期完成軌跡緩衝
        self.recent_finalized = deque(maxlen=120)

        # Debug 設定
        if debug:
            self.debug_folder = self.output_folder / "debug_frames"
            self.debug_folder.mkdir(exist_ok=True)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Log start: {datetime.now()}\n")
            f.write(f"光鬼設定: length={light_trail_length}, mode={light_trail_mode}\n")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

    # ==================== 場景檢測（光鬼版本需要更寬鬆）====================
    def is_scene_change(self, delta, threshold_ratio=0.08, debug_info=None):
        """
        光鬼版本的場景切換檢測需要更高閾值
        因為光鬼累積會讓正常畫面的差異變大
        """
        total_pixels = delta.shape[0] * delta.shape[1]
        changed_pixels = np.count_nonzero(delta > 20)  # 提高像素閾值 10→15
        change_ratio = changed_pixels / float(total_pixels)
        
        # Debug 資訊
        if debug_info is not None and self.debug:
            debug_info['change_ratio'] = change_ratio
            debug_info['threshold'] = threshold_ratio
        
        return change_ratio > threshold_ratio  # 提高比例 0.02→0.08

    # ==================== 亮點偵測（使用光鬼）====================
    def find_bright_spots(self, delta_frame, threshold=5):
        _, binary = cv2.threshold(delta_frame, int(threshold), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # 放寬面積限制（光鬼會延長軌跡）
            if 0.5 < area < 50000:
                M = cv2.moments(contour)
                if M.get("m00", 0) > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    mask = np.zeros_like(delta_frame)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    brightness = cv2.mean(delta_frame, mask=mask)[0]
                    x, y, w, h = cv2.boundingRect(contour)
                    spots.append({
                        'x': cx, 'y': cy, 'area': area,
                        'brightness': float(brightness), 'contour': contour,
                        'bbox': (x, y, w, h)
                    })
        return spots

    # ==================== 軌跡預測（Kalman）====================
    def predict_next_position(self, track, predict_ahead_seconds=None):
        if predict_ahead_seconds is None:
            predict_ahead_seconds = 1.0 / max(1.0, self.fps)
        try:
            kclone = track['kalman'].clone()
            kclone.predict(predict_ahead_seconds)
            pred = kclone.x
            return float(pred[0]), float(pred[1])
        except Exception:
            last = track['points'][-1]
            return last['x'], last['y']

    # ==================== 軌跡一致性檢查 ====================
    def check_trajectory_consistency(self, track, new_spot):
        """結合 Kalman 預測與方向一致性"""
        if len(track['points']) < 3:
            return True

        # 1. Kalman 預測檢查（放寬）
        pred_x, pred_y = self.predict_next_position(track)
        pred_distance = np.hypot(new_spot['x'] - pred_x, new_spot['y'] - pred_y)

        if 'predicted_velocity' in track:
            vx, vy = track['predicted_velocity']
            expected_speed = np.hypot(vx, vy)
            max_deviation = max(expected_speed * 2.0, 40)  # 放寬 1.5→2.0, 30→40
            if pred_distance > max_deviation:
                if self.debug:
                    self.log(f"    [預測偏差] Track {track['track_id']}: "
                            f"{pred_distance:.1f} > {max_deviation:.1f}")
                return False

        # 2. 方向一致性檢查（放寬角度）
        if len(track['points']) >= 4:
            p1 = track['points'][-2]
            p2 = track['points'][-1]
            old_vx = p2['x'] - p1['x']
            old_vy = p2['y'] - p1['y']
            new_vx = new_spot['x'] - p2['x']
            new_vy = new_spot['y'] - p2['y']

            old_angle = np.arctan2(old_vy, old_vx)
            new_angle = np.arctan2(new_vy, new_vx)
            angle_diff = abs(old_angle - new_angle) * 180 / np.pi

            if angle_diff > 60 and angle_diff < 300:  # 放寬 45→60
                if self.debug:
                    self.log(f"    [方向變化] Track {track['track_id']}: {angle_diff:.1f}°")
                return False

        return True

    # ==================== 移動檢查（寬鬆版）====================
    def check_movement(self, track):
        """檢查軌跡是否有足夠移動"""
        points = track['points']
        if len(points) < 3:  # 降低 4→3
            return True

        check_window = min(self.movement_check_frames, len(points))
        recent_points = points[-check_window:]
        start_point = recent_points[0]
        end_point = recent_points[-1]
        straight_distance = np.hypot(end_point['x'] - start_point['x'],
                                     end_point['y'] - start_point['y'])

        if straight_distance < self.min_movement_pixels:
            if self.debug:
                self.log(f"    [靜止點] Track {track['track_id']}: "
                        f"{check_window}幀僅移動 {straight_distance:.1f}px")
            return False
        
        average_speed = straight_distance / check_window
        if average_speed < 0.8:  # 大幅降低 2.0→0.8
            if self.debug:
                self.log(f"    [速度過慢] Track {track['track_id']}: "
                        f"{average_speed:.2f}px/frame")
            return False

        return True

    # ==================== 軌跡匹配 ====================
    def match_spot_to_track(self, spot, max_distance=40):  # 提高 30→40
        best_track = None
        min_score = float('inf')

        for track in self.active_tracks:
            if len(track['points']) == 0:
                continue

            pred_x, pred_y = self.predict_next_position(track)
            pred_distance = np.hypot(spot['x'] - pred_x, spot['y'] - pred_y)
            time_penalty = track['lost_frames'] * 3  # 降低懲罰 5→3
            score = pred_distance + time_penalty

            # 動態放寬距離限制
            dynamic_max = max_distance + int(np.hypot(*track['predicted_velocity']) * 2.5)

            if score < min_score and score < dynamic_max:
                if self.check_trajectory_consistency(track, spot):
                    min_score = score
                    best_track = track

        return best_track

    # ==================== 軌跡驗證（極寬鬆版，專注短軌跡）====================
    def is_meteor_track(self, track):
        """綜合驗證軌跡是否為流星 - 極寬鬆版本"""
        points = track['points']

        # 1. 最少點數（只要2點即可）
        if len(points) < 2:
            return False

        # 2. 移動檢查（大幅放寬）
        if len(points) >= 2:
            start_point = points[0]
            end_point = points[-1]
            straight_distance = np.hypot(end_point['x'] - start_point['x'],
                                        end_point['y'] - start_point['y'])
            
            # 只要移動超過2像素就算
            if straight_distance < 2.0:
                if self.debug:
                    self.log(f"  [靜止點] Track {track['track_id']}: 僅移動 {straight_distance:.1f}px")
                return False

        # 3. 場景切換排除
        start_frame = track['start_frame']
        for scene_frame in self.scene_change_frames:
            if abs(start_frame - scene_frame) <= 3:  # 降低 5→3
                return False

        # 4. 軌跡長度（極寬鬆）
        total_length = self.calculate_trajectory_length(points)
        if total_length < 3:  # 降低 5→3
            if self.debug:
                self.log(f"  [太短] Track {track['track_id']}: {total_length:.1f}px")
            return False

        # 5. 速度檢查（極寬鬆範圍）
        duration = points[-1]['frame'] - points[0]['frame']
        if duration == 0:
            duration = 1
        speed = total_length / duration

        if speed < 0.5 or speed > 100:  # 放寬 0.8-50 → 0.5-100
            if self.debug:
                self.log(f"  [速度異常] Track {track['track_id']}: {speed:.1f}px/frame")
            return False

        # 6. 短軌跡跳過線性度檢查（只對長軌跡檢查）
        if len(points) >= 5:
            pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
            try:
                [vx, vy, cx, cy] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                line_point = np.array([cx[0], cy[0]])
                line_direction = np.array([vx[0], vy[0]])
                distances = []
                for pt in pts:
                    vec = pt - line_point
                    dist = abs(np.cross(line_direction, vec))
                    distances.append(dist)
                mean_deviation = np.mean(distances)

                if mean_deviation > 50.0:  # 放寬 40→50
                    if self.debug:
                        self.log(f"  [非線性] Track {track['track_id']}: {mean_deviation:.1f}")
                    return False
            except Exception:
                pass

        # 7. 亮度檢查（極寬鬆）
        brightnesses = [p.get('brightness', 0.0) for p in points]
        max_brightness = max(brightnesses) if brightnesses else 0.0
        if max_brightness < 1.5:  # 降低 2.0→1.5
            if self.debug:
                self.log(f"  [亮度低] Track {track['track_id']}: {max_brightness:.1f}")
            return False

        return True

    # ==================== 軌跡操作 ====================
    def create_new_track(self, spot, frame_idx):
        curr_time = frame_idx / max(1.0, self.fps)
        kalman = SimpleKalman(spot['x'], spot['y'], init_vx=0.0, init_vy=0.0,
                              process_var=30.0, meas_var=10.0)  # 提高雜訊容忍
        track = {
            'track_id': self.meteor_id_counter,
            'points': [{'x': spot['x'], 'y': spot['y'], 'frame': frame_idx,
                       'brightness': spot['brightness'], 'area': spot.get('area', 0.0)}],
            'start_frame': frame_idx,
            'last_update': frame_idx,
            'lost_frames': 0,
            'predicted_velocity': (0, 0),
            'kalman': kalman,
            'last_time': curr_time
        }
        self.meteor_id_counter += 1
        return track

    def update_track(self, track, spot, frame_idx):
        curr_time = frame_idx / max(1.0, self.fps)
        dt = curr_time - track.get('last_time', curr_time)
        if dt <= 0:
            dt = 1.0 / max(1.0, self.fps)

        track['kalman'].predict(dt)
        kstate = track['kalman'].update(spot['x'], spot['y'])
        track['predicted_velocity'] = (float(kstate[2]), float(kstate[3]))

        track['points'].append({
            'x': int(kstate[0]), 'y': int(kstate[1]), 'frame': frame_idx,
            'brightness': spot['brightness'], 'area': spot.get('area', 0.0)
        })
        track['last_update'] = frame_idx
        track['lost_frames'] = 0
        track['last_time'] = curr_time

    # ==================== 流星合併機制 ====================
    def should_merge_meteors(self, meteor1, meteor2):
        """判斷兩個流星是否應該合併為一條"""
        # 1. 時間重疊或接近
        r1 = meteor1['frame_range']
        r2 = meteor2['frame_range']
        time_gap = max(0, r2[0] - r1[1], r1[0] - r2[1])
        
        if time_gap > 30:  # 時間間隔超過30幀不合併
            return False
        
        # 2. 角度相似（方向一致）
        angle1 = meteor1['angle']
        angle2 = meteor2['angle']
        angle_diff = abs(angle1 - angle2)
        # 處理角度循環（例如 -179° 和 179° 其實很接近）
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        if angle_diff > 30:  # 角度差異超過30度不合併
            return False
        
        # 3. 軌跡距離（檢查兩條軌跡是否接近）
        traj1 = meteor1['trajectory']
        traj2 = meteor2['trajectory']
        
        # 計算端點距離
        distances = [
            np.hypot(traj1[-1]['x'] - traj2[0]['x'], traj1[-1]['y'] - traj2[0]['y']),  # 1的尾 到 2的頭
            np.hypot(traj2[-1]['x'] - traj1[0]['x'], traj2[-1]['y'] - traj1[0]['y']),  # 2的尾 到 1的頭
            np.hypot(traj1[0]['x'] - traj2[0]['x'], traj1[0]['y'] - traj2[0]['y']),    # 頭對頭
            np.hypot(traj1[-1]['x'] - traj2[-1]['x'], traj1[-1]['y'] - traj2[-1]['y']) # 尾對尾
        ]
        min_distance = min(distances)
        
        # 動態距離閾值（根據速度調整）
        avg_speed = (meteor1['speed'] + meteor2['speed']) / 2
        max_distance = max(50, avg_speed * 10)  # 至少50像素，或速度*10
        
        if min_distance > max_distance:
            return False
        
        # 4. 速度相似
        speed_ratio = min(meteor1['speed'], meteor2['speed']) / max(meteor1['speed'], meteor2['speed'], 0.01)
        if speed_ratio < 0.5:  # 速度差異超過2倍不合併
            return False
        
        return True
    
    def merge_two_meteors(self, meteor1, meteor2):
        """合併兩個流星成一條完整軌跡"""
        # 合併軌跡點（按時間排序）
        all_points = meteor1['trajectory'] + meteor2['trajectory']
        all_points.sort(key=lambda p: p['frame'])
        
        # 去除重複幀（保留亮度較高的）
        unique_points = []
        prev_frame = -1
        for point in all_points:
            if point['frame'] != prev_frame:
                unique_points.append(point)
                prev_frame = point['frame']
            elif point['brightness'] > unique_points[-1]['brightness']:
                unique_points[-1] = point
        
        # 計算合併後的屬性
        xs = [p['x'] for p in unique_points]
        ys = [p['y'] for p in unique_points]
        
        merged = {
            'meteor_id': meteor1['meteor_id'],  # 保留第一個ID
            'frame_range': [unique_points[0]['frame'], unique_points[-1]['frame']],
            'duration_frames': len(unique_points),
            'trajectory': unique_points,
            'bbox': [max(0, min(xs) - 10), max(0, min(ys) - 10),
                    max(xs) - min(xs) + 20, max(ys) - min(ys) + 20],
            'length': self.calculate_trajectory_length(unique_points),
            'angle': self.calculate_trajectory_angle(unique_points),
            'speed': self.calculate_average_speed(unique_points),
            'max_brightness': max([p.get('brightness', 0.0) for p in unique_points])
        }
        
        return merged
    
    def merge_duplicate_meteors(self):
        """合併重複偵測的流星"""
        if len(self.meteors) < 2:
            return
        
        merged_count = 0
        changed = True
        
        while changed:
            changed = False
            i = 0
            while i < len(self.meteors):
                j = i + 1
                while j < len(self.meteors):
                    if self.should_merge_meteors(self.meteors[i], self.meteors[j]):
                        # 合併 i 和 j
                        merged = self.merge_two_meteors(self.meteors[i], self.meteors[j])
                        self.meteors[i] = merged
                        del self.meteors[j]
                        merged_count += 1
                        changed = True
                        if self.debug:
                            self.log(f"  [合併] 合併流星 {i} 和 {j} -> 新長度 {merged['length']:.1f}px")
                        break
                    j += 1
                i += 1
        
        if merged_count > 0:
            self.log(f"✓ 合併了 {merged_count} 組重複流星")
            # 重新編號
            for idx, meteor in enumerate(self.meteors):
                meteor['meteor_id'] = f"meteor_{idx:03d}"
    def calculate_trajectory_length(self, points):
        length = 0.0
        for i in range(1, len(points)):
            dx = points[i]['x'] - points[i-1]['x']
            dy = points[i]['y'] - points[i-1]['y']
            length += np.hypot(dx, dy)
        return float(length)

    def calculate_trajectory_angle(self, points):
        if len(points) < 2:
            return 0.0
        dx = points[-1]['x'] - points[0]['x']
        dy = points[-1]['y'] - points[0]['y']
        return float(np.arctan2(dy, dx) * 180.0 / np.pi)

    def calculate_average_speed(self, points):
        length = self.calculate_trajectory_length(points)
        duration = points[-1]['frame'] - points[0]['frame']
        if duration == 0:
            duration = 1
        return float(length / duration)

    def finalize_track(self, track):
        """完成軌跡並轉換為流星資料"""
        if not self.is_meteor_track(track):
            return None

        points = track['points']
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]

        meteor = {
            'meteor_id': f"meteor_{track['track_id']:03d}",
            'frame_range': [track['start_frame'], points[-1]['frame']],
            'duration_frames': len(points),
            'trajectory': points,
            'bbox': [max(0, min(xs) - 10), max(0, min(ys) - 10),
                    max(xs) - min(xs) + 20, max(ys) - min(ys) + 20],
            'length': self.calculate_trajectory_length(points),
            'angle': self.calculate_trajectory_angle(points),
            'speed': self.calculate_average_speed(points),
            'max_brightness': max([p.get('brightness', 0.0) for p in points])
        }

        return meteor

    # ==================== Debug 輸出（加入光鬼視覺化）====================
    def save_debug_frame(self, frame, delta, light_trail_delta, spots, frame_idx):
        if not self.debug or frame_idx % 25 != 0:
            return

        vis = frame.copy()
        delta_color = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
        trail_color = cv2.cvtColor(light_trail_delta, cv2.COLOR_GRAY2BGR)

        # 標記偵測到的亮點
        for spot in spots:
            cv2.circle(vis, (spot['x'], spot['y']), 5, (0, 255, 0), 2)

        # 繪製活躍軌跡
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255), (128, 255, 0)]
        for idx, track in enumerate(self.active_tracks):
            color = colors[idx % len(colors)]
            points = track['points']
            for i in range(1, len(points)):
                p1 = (points[i-1]['x'], points[i-1]['y'])
                p2 = (points[i]['x'], points[i]['y'])
                cv2.line(vis, p1, p2, color, 2)

            if len(points) >= 2:
                pred_x, pred_y = self.predict_next_position(track)
                cv2.circle(vis, (int(pred_x), int(pred_y)), 3, color, -1)

        # 三欄顯示：原圖 | 標準差分 | 光鬼差分
        combined = np.hstack([vis, delta_color, trail_color])
        info = (f"Frame {frame_idx} | Spots: {len(spots)} | "
               f"Active: {len(self.active_tracks)} | Light Trail: {self.light_trail_mode}")
        cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        filename = self.debug_folder / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(filename), combined)

    # ==================== 主要處理流程（整合光鬼）====================
    def process_video(self):
        """處理整個影片"""
        self.log("=" * 80)
        self.log("光鬼增強版流星偵測系統")
        self.log("=" * 80)
        self.log(f"影片: {self.video_path}")
        self.log(f"總幀數: {self.total_frames}, FPS: {self.fps:.2f}")
        self.log(f"光鬼設定: length={self.light_trail.trail_length}, "
                f"mode={self.light_trail_mode}")
        self.log(f"參數: movement_check={self.movement_check_frames}幀, "
                f"min_movement={self.min_movement_pixels}px")
        if self.debug:
            self.log("調試模式: 開啟")

        cap = cv2.VideoCapture(str(self.video_path))
        ret, first_frame = cap.read()
        if not ret:
            self.log("❌ 無法讀取影片")
            return []

        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (3, 3), 0)
        self.gray_buffer.append(prev_gray)
        self.light_trail.add_frame(prev_gray)

        frame_idx = 1

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0)
            self.gray_buffer.append(curr_gray)
            self.light_trail.add_frame(curr_gray)

            # === 生成光鬼效果 ===
            if self.light_trail_mode == 'weighted':
                trail_gray = self.light_trail.get_light_trail()
            elif self.light_trail_mode == 'max':
                trail_gray = self.light_trail.get_max_projection()
            else:  # enhanced
                trail_gray = self.light_trail.get_enhanced_trail()

            # Multi-frame 差分（標準方式）
            N = max(1, int(round(self.target_window_seconds * self.fps)))
            if len(self.gray_buffer) >= N:
                prev_gray_k = self.gray_buffer[-N]
            else:
                prev_gray_k = self.gray_buffer[0]

            delta_standard = cv2.absdiff(curr_gray, prev_gray_k)
            delta_standard = cv2.medianBlur(delta_standard, 3)

            # 光鬼差分（與光鬼影像比較）
            if trail_gray is not None and len(self.gray_buffer) >= self.light_trail.trail_length:
                # 計算當前幀與光鬼影像的差異
                delta_trail = cv2.absdiff(curr_gray, trail_gray)
                delta_trail = cv2.medianBlur(delta_trail, 3)
                
                # 結合兩種差分（取較強的信號）
                delta = cv2.max(delta_standard, delta_trail)
            else:
                delta = delta_standard

            # 場景切換檢查（使用標準差分避免光鬼誤判）
            scene_debug = {}
            is_scene_change = self.is_scene_change(delta_standard, debug_info=scene_debug)
            if is_scene_change:
                self.scene_change_frames.append(frame_idx)
                if self.debug:
                    self.log(f"  [場景切換] Frame {frame_idx} - "
                            f"變化率: {scene_debug.get('change_ratio', 0):.3f} > "
                            f"{scene_debug.get('threshold', 0):.3f}")
                self.active_tracks.clear()
                self.light_trail.frame_buffer.clear()  # 清空光鬼緩衝
                prev_gray = curr_gray
                frame_idx += 1
                continue

            # 動態閾值（降低以捕捉微弱流星）
            dynamic_threshold = max(6, int(10 * np.sqrt(max(1, N))))
            spots = self.find_bright_spots(delta, threshold=dynamic_threshold)

            if self.debug and (len(spots) > 0 or len(self.active_tracks) > 0):
                self.save_debug_frame(curr_frame, delta_standard, delta, 
                                     spots, frame_idx)

            # 更新軌跡遺失計數
            for track in self.active_tracks:
                track['lost_frames'] += 1

            # 匹配點到軌跡
            matched_tracks = set()
            for spot in spots:
                track = self.match_spot_to_track(spot)
                if track is not None and track['track_id'] not in matched_tracks:
                    self.update_track(track, spot, frame_idx)
                    matched_tracks.add(track['track_id'])
                else:
                    new_track = self.create_new_track(spot, frame_idx)
                    self.active_tracks.append(new_track)

            # 檢查完成的軌跡
            completed_tracks = []
            for track in self.active_tracks[:]:
                if track['lost_frames'] > self.lost_track_threshold:
                    self.active_tracks.remove(track)
                    completed_tracks.append(track)

            # 驗證並儲存流星
            for track in completed_tracks:
                meteor = self.finalize_track(track)
                if meteor is not None:
                    self.meteors.append(meteor)
                    self.log(f"✓ 流星 #{len(self.meteors)}: "
                            f"幀 {meteor['frame_range'][0]}-{meteor['frame_range'][1]} "
                            f"(時間 {meteor['frame_range'][0]/self.fps:.1f}s-"
                            f"{meteor['frame_range'][1]/self.fps:.1f}s), "
                            f"長度 {meteor['length']:.1f}px, "
                            f"速度 {meteor['speed']:.1f}px/frame, "
                            f"點數 {meteor['duration_frames']}")

            frame_idx += 1
            if frame_idx % 100 == 0:
                self.log(f"處理進度: {frame_idx}/{self.total_frames}, "
                        f"活躍軌跡: {len(self.active_tracks)}, "
                        f"已偵測流星: {len(self.meteors)}")

        # 處理剩餘軌跡
        for track in self.active_tracks:
            meteor = self.finalize_track(track)
            if meteor is not None:
                self.meteors.append(meteor)

        cap.release()

        # === 合併重複偵測的流星 ===
        if len(self.meteors) > 0:
            self.log("\n開始合併重複偵測的流星...")
            before_merge = len(self.meteors)
            self.merge_duplicate_meteors()
            after_merge = len(self.meteors)
            if before_merge != after_merge:
                self.log(f"合併前: {before_merge} 顆 → 合併後: {after_merge} 顆")

        self.log(f"\n總共偵測到 {len(self.meteors)} 顆流星")
        self.log(f"場景切換次數: {len(self.scene_change_frames)}")

        self.save_results()
        if len(self.meteors) > 0:
            self.create_annotated_video()
        else:
            self.log("⚠️ 未偵測到流星")
            if not self.debug:
                self.log("建議使用 --debug 參數重新執行以查看偵測細節")

        return self.meteors

    # ==================== 結果儲存 ====================
    def save_results(self):
        """儲存偵測結果為 JSON"""
        result = {
            'video': str(self.video_path),
            'total_meteors': len(self.meteors),
            'fps': self.fps,
            'light_trail_config': {
                'length': self.light_trail.trail_length,
                'mode': self.light_trail_mode
            },
            'scene_changes': self.scene_change_frames,
            'meteors': self.meteors
        }

        with open(self.output_folder / "results.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.log(f"✓ 結果已儲存: {self.output_folder / 'results.json'}")

    # ==================== 標記影片生成 ====================
    def create_annotated_video(self):
        """建立標記後的影片"""
        self.log("\n開始生成標記影片...")

        output_video = self.output_folder / f"{self.video_path.stem}_annotated.mp4"

        cap = cv2.VideoCapture(str(self.video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, self.fps,
                             (self.width, self.height))

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 標記場景切換
            if frame_idx in self.scene_change_frames:
                cv2.putText(frame, "SCENE CHANGE", (self.width//2-100, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # 標記所有流星
            for meteor in self.meteors:
                start_f, end_f = meteor['frame_range']
                if start_f <= frame_idx <= end_f:
                    # 繪製完整軌跡
                    points = meteor['trajectory']
                    for i in range(1, len(points)):
                        p1 = (points[i-1]['x'], points[i-1]['y'])
                        p2 = (points[i]['x'], points[i]['y'])
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)

                    # 標記當前位置
                    for point in points:
                        if point['frame'] == frame_idx:
                            cv2.circle(frame, (point['x'], point['y']),
                                       8, (0, 255, 255), 2)
                            cv2.circle(frame, (point['x'], point['y']),
                                       3, (0, 0, 255), -1)

                    # 繪製邊界框
                    bbox = meteor['bbox']
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                  (255, 0, 0), 1)

                    # 標註資訊
                    label = f"{meteor['meteor_id']} | {meteor['speed']:.1f}px/f"
                    cv2.putText(frame, label,
                                (bbox[0], max(15, bbox[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 加入幀號和時間
            time_sec = frame_idx / self.fps
            info = f"Frame: {frame_idx} | Time: {time_sec:.1f}s | Meteors: {len(self.meteors)}"
            cv2.putText(frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                self.log(f"  標記進度: {frame_idx}/{self.total_frames}")

        cap.release()
        out.release()

        self.log(f"✓ 標記影片已儲存: {output_video}")
        return output_video


# ==================== 命令列介面 ====================
def main():
    parser = argparse.ArgumentParser(
        description='光鬼增強版流星偵測系統 - Light Trail + Kalman',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用（預設光鬼長度5幀）
  python meteor_detector_light_trail.py --video me.mp4
  
  # 調整光鬼長度（3-8幀建議範圍）
  python meteor_detector_light_trail.py --video me.mp4 --trail-length 7
  
  # 選擇光鬼模式
  python meteor_detector_light_trail.py --video me.mp4 --trail-mode max
  
  # 開啟調試模式
  python meteor_detector_light_trail.py --video me.mp4 --debug
  
光鬼模式說明:
  - weighted: 加權平均（平滑，適合慢速流星）
  - max: 最大值投影（保留最亮像素，適合快速流星）
  - enhanced: 混合模式（預設，兼顧兩者優點）
        """
    )
    parser.add_argument('--video', type=str, required=True, help='影片路徑')
    parser.add_argument('--trail-length', type=int, default=5, 
                       help='光鬼累積幀數 (建議 3-8，預設 5)')
    parser.add_argument('--trail-mode', type=str, default='enhanced',
                       choices=['weighted', 'max', 'enhanced'],
                       help='光鬼模式 (預設 enhanced)')
    parser.add_argument('--debug', action='store_true', 
                       help='啟用調試模式（輸出偵測過程圖片）')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ 找不到影片: {args.video}")
        return

    output_folder = Path("output_light_trail") / Path(args.video).stem
    tracker = MeteorTrackerWithLightTrail(
        args.video, 
        output_folder, 
        debug=args.debug,
        light_trail_length=args.trail_length,
        light_trail_mode=args.trail_mode
    )
    tracker.process_video()


if __name__ == '__main__':
    main()