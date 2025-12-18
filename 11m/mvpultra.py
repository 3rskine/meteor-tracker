#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光軌增強版流星偵測系統 - 進階優化版
新增功能：
1. 多層級偵測（同時用不同參數偵測）
2. 恆星過濾器（背景減除 + 靜止點過濾）
3. 場景切換保護增強
4. 軌跡品質評分系統
5. 自適應參數調整
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


# ==================== 恆星背景建模器 ====================
class StarBackgroundModel:
    """用於識別和過濾靜止的恆星"""
    
    def __init__(self, buffer_size=30, stability_threshold=3):
        self.position_history = defaultdict(lambda: deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.stability_threshold = stability_threshold
        
    def update(self, spots, frame_idx):
        """更新每個 spot 的位置歷史"""
        current_positions = set()
        
        for spot in spots:
            key = (spot['x'] // 5, spot['y'] // 5)  # 網格化，允許小範圍抖動
            current_positions.add(key)
            self.position_history[key].append(frame_idx)
    
    def is_static_star(self, spot, frame_idx, min_appearances=15):
        """判斷是否為靜止恆星（在同一位置持續出現）"""
        key = (spot['x'] // 5, spot['y'] // 5)
        
        if key not in self.position_history:
            return False
        
        history = list(self.position_history[key])
        if len(history) < min_appearances:
            return False
        
        # 檢查是否連續出現（允許少量間斷）
        recent_frames = [f for f in history if frame_idx - f < self.buffer_size]
        if len(recent_frames) >= min_appearances:
            gaps = [recent_frames[i+1] - recent_frames[i] for i in range(len(recent_frames)-1)]
            avg_gap = np.mean(gaps) if gaps else 0
            return avg_gap < 3  # 幾乎每幀都出現
        
        return False
    
    def cleanup_old_entries(self, current_frame, max_age=60):
        """清理舊的記錄"""
        keys_to_remove = []
        for key, frames in self.position_history.items():
            if frames and current_frame - frames[-1] > max_age:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.position_history[key]


# ==================== 光軌處理器 ====================
class LightTrailProcessor:
    """光軌 (Light Trail) 處理器 - 累積移動物體軌跡"""

    def __init__(self, trail_length=10, decay_type='uniform'):
        self.trail_length = trail_length
        self.decay_type = decay_type
        self.frame_buffer = deque(maxlen=trail_length)
        self.weights = self._compute_weights()

    def _compute_weights(self):
        n = max(1, self.trail_length)
        if self.decay_type == 'exponential':
            weights = np.exp(-np.arange(n) * 0.35)
        elif self.decay_type == 'linear':
            weights = np.linspace(0.3, 1.0, n)
        else:
            weights = np.ones(n)
        weights = weights / weights.sum()
        return weights[::-1]

    def add_frame(self, frame):
        self.frame_buffer.append(frame.copy())

    def get_light_trail(self):
        if len(self.frame_buffer) == 0:
            return None
        result = np.zeros_like(self.frame_buffer[0], dtype=np.float32)
        for i, frame in enumerate(self.frame_buffer):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            result += frame.astype(np.float32) * weight
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_max_projection(self):
        if len(self.frame_buffer) == 0:
            return None
        result = np.zeros_like(self.frame_buffer[0], dtype=np.uint8)
        for frame in self.frame_buffer:
            result = np.maximum(result, frame)
        return result

    def get_enhanced_trail(self):
        if len(self.frame_buffer) == 0:
            return None
        weighted = self.get_light_trail()
        max_proj = self.get_max_projection()
        enhanced = cv2.addWeighted(weighted, 0.7, max_proj, 0.3, 0)
        return enhanced


# ==================== 軌跡品質評分器 ====================
class TrajectoryQualityScorer:
    """評估軌跡品質，區分流星與雜訊"""
    
    @staticmethod
    def calculate_linearity_score(points):
        """線性度評分 (0-1)，1 表示完美直線"""
        if len(points) < 3:
            return 0.5
        
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
            # 轉換為 0-1 分數，偏差越小分數越高
            score = 1.0 / (1.0 + mean_deviation / 10.0)
            return float(score)
        except:
            return 0.5
    
    @staticmethod
    def calculate_velocity_consistency(points):
        """速度一致性評分 (0-1)"""
        if len(points) < 3:
            return 0.5
        
        velocities = []
        for i in range(1, len(points)):
            dx = points[i]['x'] - points[i-1]['x']
            dy = points[i]['y'] - points[i-1]['y']
            v = np.hypot(dx, dy)
            velocities.append(v)
        
        if not velocities:
            return 0.5
        
        mean_v = np.mean(velocities)
        std_v = np.std(velocities)
        
        # 變異係數越小，一致性越高
        cv = std_v / (mean_v + 1e-6)
        score = 1.0 / (1.0 + cv)
        return float(score)
    
    @staticmethod
    def calculate_brightness_profile(points):
        """亮度變化評分 (0-1)，流星應該有亮度高峰"""
        if len(points) < 2:
            return 0.5
        
        brightnesses = [p.get('brightness', 0) for p in points]
        max_b = max(brightnesses)
        min_b = min(brightnesses)
        
        # 亮度對比
        contrast = (max_b - min_b) / (max_b + 1e-6)
        
        # 檢查是否有明顯峰值
        mid_idx = len(brightnesses) // 2
        mid_brightness = np.mean(brightnesses[max(0, mid_idx-1):min(len(brightnesses), mid_idx+2)])
        edge_brightness = (brightnesses[0] + brightnesses[-1]) / 2
        
        peak_score = mid_brightness / (edge_brightness + 1e-6)
        peak_score = min(1.0, peak_score / 2.0)
        
        return float(contrast * 0.6 + peak_score * 0.4)
    
    @classmethod
    def calculate_overall_score(cls, track):
        """綜合評分 (0-100)"""
        points = track['points']
        
        if len(points) < 2:
            return 0.0
        
        # 各項子分數
        linearity = cls.calculate_linearity_score(points)
        velocity_consistency = cls.calculate_velocity_consistency(points)
        brightness = cls.calculate_brightness_profile(points)
        
        # 長度獎勵
        length = sum(np.hypot(points[i]['x'] - points[i-1]['x'],
                             points[i]['y'] - points[i-1]['y'])
                    for i in range(1, len(points)))
        length_score = min(1.0, length / 50.0)  # 50px 以上給滿分
        
        # 持續時間獎勵
        duration = len(points)
        duration_score = min(1.0, duration / 5.0)  # 5 幀以上給滿分
        
        # 加權總分
        total_score = (
            linearity * 30 +
            velocity_consistency * 25 +
            brightness * 20 +
            length_score * 15 +
            duration_score * 10
        )
        
        return float(total_score)


# ==================== 主要偵測類別 ====================
class AdvancedMeteorTracker:
    def __init__(self, video_path, output_folder, debug=False,
                 light_trail_length=3, light_trail_mode='enhanced', 
                 proc_scale=1.0, sensitivity='medium'):
        self.video_path = Path(video_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.proc_scale = float(proc_scale) if proc_scale is not None else 1.0
        if self.proc_scale <= 0 or self.proc_scale > 1.0:
            self.proc_scale = 1.0

        # 讀取影片資訊
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片: {video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        cap.release()

        # === 場景切換保護增強 ===
        self.scene_change_buffer = deque(maxlen=10)  # 記錄最近的場景切換
        self.scene_cooldown_frames = 8  # 場景切換後的冷卻期（幀數）

        # 光軌處理器
        self.light_trail = LightTrailProcessor(
            trail_length=light_trail_length,
            decay_type='linear'
        )
        self.light_trail_mode = light_trail_mode

        # **新增：恆星背景模型**
        self.star_background = StarBackgroundModel(buffer_size=30, stability_threshold=3)

        # **新增：品質評分器**
        self.quality_scorer = TrajectoryQualityScorer()

        # 偵測結果
        self.meteors = []
        self.rejected_tracks = []  # 儲存被拒絕的軌跡供分析
        self.log_file = self.output_folder / "detection.log"
        self.active_tracks = []
        self.meteor_id_counter = 0
        self.scene_change_frames = []

        # **敏感度預設**
        self.sensitivity = sensitivity
        self._configure_sensitivity(sensitivity)

        # Multi-frame 差分設定
        self.target_window_seconds = 0.15
        self.gray_buffer = deque(maxlen=max(60, int(self.fps * 2)))

        # 近期完成軌跡緩衝 & pending spots
        self.recent_finalized = deque(maxlen=120)
        self.pending_spots = deque(maxlen=8)

        # morphology / area thresholds
        self.morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Debug 設定
        if debug:
            self.debug_folder = self.output_folder / "debug_frames"
            self.debug_folder.mkdir(exist_ok=True)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Log start: {datetime.now()}\n")
            f.write(f"進階偵測模式: sensitivity={sensitivity}\n")
            f.write(f"光軌設定: length={light_trail_length}, mode={light_trail_mode}, proc_scale={self.proc_scale}\n")

    def _configure_sensitivity(self, level):
        """根據敏感度等級配置參數"""
        if level == 'high':
            # 高敏感度：容易偵測但誤報多
            self.min_movement_pixels = 1
            self.movement_check_frames = 2
            self.lost_track_threshold = 12
            self.min_area_pixels = 2
            self.spot_frame_cap = 300
            self.min_max_brightness = 1.0
            self.min_speed_px_per_frame = 0.5
            self.quality_threshold = 35  # 較低的品質門檻
            self.base_threshold = 6
            self.min_track_duration = 3  # 最少持續 3 幀
            
        elif level == 'low':
            # 低敏感度：只抓明顯流星
            self.min_movement_pixels = 3
            self.movement_check_frames = 3
            self.lost_track_threshold = 8
            self.min_area_pixels = 5
            self.spot_frame_cap = 150
            self.min_max_brightness = 2.5
            self.min_speed_px_per_frame = 2.0
            self.quality_threshold = 60  # 較高的品質門檻
            self.base_threshold = 12
            self.min_track_duration = 4  # 最少持續 4 幀
            
        else:  # medium (default)
            self.min_movement_pixels = 2
            self.movement_check_frames = 2
            self.lost_track_threshold = 10
            self.min_area_pixels = 3
            self.spot_frame_cap = 200
            self.min_max_brightness = 1.5
            self.min_speed_px_per_frame = 1.0
            self.quality_threshold = 45  # 中等品質門檻
            self.base_threshold = 8
            self.min_track_duration = 3  # **最少持續 3 幀**（核心過濾）

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

    def is_scene_change(self, delta, curr_gray, prev_gray, threshold_ratio=0.06, debug_info=None):
        """多重檢測機制判斷場景切換"""
        total_pixels = delta.shape[0] * delta.shape[1]
        
        # 方法1: 變化像素比例
        changed_pixels = np.count_nonzero(delta > 20)
        change_ratio = changed_pixels / float(total_pixels)
        
        # 方法2: 直方圖差異 (檢測整體亮度/色調變化)
        hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
        hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
        hist_diff = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_BHATTACHARYYA)
        
        # 方法3: 平均亮度突變
        mean_curr = np.mean(curr_gray)
        mean_prev = np.mean(prev_gray)
        brightness_change = abs(mean_curr - mean_prev)
        
        # 方法4: 結構相似度（SSIM 簡化版）
        # 計算相關係數
        correlation = np.corrcoef(curr_gray.flatten(), prev_gray.flatten())[0, 1]
        
        is_scene = False
        reasons = []
        
        # 多重條件判斷
        if change_ratio > threshold_ratio:
            is_scene = True
            reasons.append(f"像素變化 {change_ratio:.3f}")
        
        if hist_diff > 0.5:  # Bhattacharyya 距離 > 0.5 表示差異大
            is_scene = True
            reasons.append(f"直方圖差異 {hist_diff:.3f}")
        
        if brightness_change > 30:  # 平均亮度變化超過 30
            is_scene = True
            reasons.append(f"亮度突變 {brightness_change:.1f}")
        
        if correlation < 0.5:  # 相關係數過低
            is_scene = True
            reasons.append(f"結構相似度低 {correlation:.3f}")

        if debug_info is not None:
            debug_info['change_ratio'] = change_ratio
            debug_info['hist_diff'] = hist_diff
            debug_info['brightness_change'] = brightness_change
            debug_info['correlation'] = correlation
            debug_info['reasons'] = reasons
            debug_info['threshold'] = threshold_ratio

        return is_scene, reasons

    def find_bright_spots(self, delta_frame, threshold=2, scale_up_factor=1.5):
        _, binary = cv2.threshold(delta_frame, int(threshold), 255, cv2.THRESH_BINARY)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_open_kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.morph_close_kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area_pixels < area < 50000:
                M = cv2.moments(contour)
                if M.get("m00", 0) > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    mask = np.zeros_like(delta_frame)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    brightness = cv2.mean(delta_frame, mask=mask)[0]
                    x, y, w, h = cv2.boundingRect(contour)

                    if scale_up_factor != 1.0:
                        inv = 1.0 / scale_up_factor
                        cx = int(round(cx * inv))
                        cy = int(round(cy * inv))
                        x = int(round(x * inv))
                        y = int(round(y * inv))
                        w = int(round(w * inv))
                        h = int(round(h * inv))

                    spots.append({
                        'x': cx, 'y': cy, 'area': area,
                        'brightness': float(brightness), 'contour': contour,
                        'bbox': (x, y, w, h)
                    })
        return spots
    
    def is_in_scene_cooldown(self, frame_idx):
        """檢查是否在場景切換冷卻期內"""
        for scene_frame in self.scene_change_buffer:
            if abs(frame_idx - scene_frame) <= self.scene_cooldown_frames:
                return True
        return False

    def stable_spot_exists(self, spot, radius=6, required_frames=2):
        if required_frames <= 1:
            return True
        count = 0
        recent = list(self.pending_spots)
        if len(recent) < required_frames:
            return False
        for frame_spots in recent[-required_frames:]:
            if any(np.hypot(spot['x'] - x, spot['y'] - y) <= radius for x, y in frame_spots):
                count += 1
        return count >= required_frames

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

    def check_trajectory_consistency(self, track, new_spot):
        if len(track['points']) < 3:
            return True

        pred_x, pred_y = self.predict_next_position(track)
        pred_distance = np.hypot(new_spot['x'] - pred_x, new_spot['y'] - pred_y)

        if 'predicted_velocity' in track:
            vx, vy = track['predicted_velocity']
            expected_speed = np.hypot(vx, vy)
            max_deviation = max(expected_speed * 2.0, 40)
            if pred_distance > max_deviation:
                return False

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

            if angle_diff > 60 and angle_diff < 300:
                return False

        return True

    def check_movement(self, track):
        points = track['points']
        if len(points) < 3:
            return True

        check_window = min(self.movement_check_frames, len(points))
        recent_points = points[-check_window:]
        start_point = recent_points[0]
        end_point = recent_points[-1]
        straight_distance = np.hypot(end_point['x'] - start_point['x'],
                                     end_point['y'] - start_point['y'])

        if straight_distance < self.min_movement_pixels:
            return False

        average_speed = straight_distance / check_window
        if average_speed < 2.5:
            return False

        return True

    def match_spot_to_track(self, spot, max_distance=100):
        best_track = None
        min_score = float('inf')

        for track in self.active_tracks:
            if len(track['points']) == 0:
                continue

            pred_x, pred_y = self.predict_next_position(track)
            pred_distance = np.hypot(spot['x'] - pred_x, spot['y'] - pred_y)
            time_penalty = track['lost_frames'] * 3
            score = pred_distance + time_penalty

            dynamic_max = max_distance + int(np.hypot(*track['predicted_velocity']) * 2.5)

            if score < min_score and score < dynamic_max:
                if self.check_trajectory_consistency(track, spot):
                    min_score = score
                    best_track = track

        return best_track

    def is_meteor_track(self, track):
        """使用品質評分系統驗證軌跡"""
        points = track['points']
        if len(points) < 2:
            return False, "點數太少"

        # **關鍵過濾：最小持續時間（幀數）**
        duration_frames = points[-1]['frame'] - points[0]['frame'] + 1
        if duration_frames < self.min_track_duration:
            return False, f"持續時間太短 ({duration_frames}幀 < {self.min_track_duration}幀)"

        # 基本檢查
        start_point = points[0]
        end_point = points[-1]
        straight_distance = np.hypot(end_point['x'] - start_point['x'],
                                    end_point['y'] - start_point['y'])
        if straight_distance < 2.0:
            return False, f"靜止點 ({straight_distance:.1f}px)"

        # **場景切換檢查加強：不只檢查起始幀，也檢查整個軌跡**
        start_frame = track['start_frame']
        end_frame = points[-1]['frame']
        
        for scene_frame in self.scene_change_buffer:
            # 如果軌跡與任何場景切換重疊或接近
            if (start_frame - self.scene_cooldown_frames <= scene_frame <= end_frame + self.scene_cooldown_frames):
                return False, f"場景切換附近 (scene@{scene_frame})"

        # 計算品質分數
        quality_score = self.quality_scorer.calculate_overall_score(track)
        track['quality_score'] = quality_score

        if quality_score < self.quality_threshold:
            return False, f"品質分數太低 ({quality_score:.1f})"

        # 速度檢查
        total_length = self.calculate_trajectory_length(points)
        duration = points[-1]['frame'] - points[0]['frame']
        if duration == 0:
            duration = 1
        speed = total_length / duration

        if speed < self.min_speed_px_per_frame or speed > 1000:
            return False, f"速度異常 ({speed:.1f}px/f)"

        # 亮度檢查
        brightnesses = [p.get('brightness', 0.0) for p in points]
        max_brightness = max(brightnesses) if brightnesses else 0.0
        if max_brightness < self.min_max_brightness:
            return False, f"亮度過低 ({max_brightness:.1f})"

        return True, f"通過 (品質: {quality_score:.1f}, 持續: {duration_frames}幀)"

    def create_new_track(self, spot, frame_idx):
        curr_time = frame_idx / max(1.0, self.fps)
        kalman = SimpleKalman(spot['x'], spot['y'], init_vx=0.0, init_vy=0.0,
                              process_var=30.0, meas_var=10.0)
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
        is_valid, reason = self.is_meteor_track(track)
        
        if not is_valid:
            # 儲存被拒絕的軌跡供分析
            track['reject_reason'] = reason
            self.rejected_tracks.append(track)
            if self.debug:
                self.log(f"    [拒絕] Track {track['track_id']}: {reason}")
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
            'max_brightness': max([p.get('brightness', 0.0) for p in points]),
            'quality_score': track.get('quality_score', 0)
        }

        return meteor

    def save_debug_frame(self, frame, delta, light_trail_delta, spots, frame_idx):
        if not self.debug or frame_idx % 25 != 0:
            return

        vis = frame.copy()
        delta_color = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
        trail_color = cv2.cvtColor(light_trail_delta, cv2.COLOR_GRAY2BGR)

        # 標記 spots（綠色=移動，紅色=靜止恆星）
        for spot in spots:
            if self.star_background.is_static_star(spot, frame_idx):
                cv2.circle(vis, (spot['x'], spot['y']), 5, (0, 0, 255), 2)  # 紅色
            else:
                cv2.circle(vis, (spot['x'], spot['y']), 5, (0, 255, 0), 2)  # 綠色

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

        combined = np.hstack([vis, delta_color, trail_color])
        info = (f"Frame {frame_idx} | Spots: {len(spots)} | "
               f"Active: {len(self.active_tracks)} | Mode: {self.light_trail_mode} | Sens: {self.sensitivity}")
        cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)

        filename = self.debug_folder / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(filename), combined)

    def process_video(self):
        self.log("=" * 80)
        self.log("進階流星偵測系統（多層級 + 恆星過濾 + 品質評分）")
        self.log("=" * 80)
        self.log(f"影片: {self.video_path}")
        self.log(f"總幀數: {self.total_frames}, FPS: {self.fps:.2f}")
        self.log(f"敏感度: {self.sensitivity} (品質門檻: {self.quality_threshold}, 最小持續: {self.min_track_duration}幀)")
        self.log(f"光軌設定: length={self.light_trail.trail_length}, mode={self.light_trail_mode}")
        if self.debug:
            self.log("調試模式: 開啟")

        cap = cv2.VideoCapture(str(self.video_path))
        ret, first_frame = cap.read()
        if not ret:
            self.log("❌ 無法讀取影片")
            return []

        prev_gray_full = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        prev_gray_full = cv2.GaussianBlur(prev_gray_full, (3, 3), 0)

        if self.proc_scale != 1.0:
            prev_gray = cv2.resize(prev_gray_full, (0, 0), fx=self.proc_scale, fy=self.proc_scale)
        else:
            prev_gray = prev_gray_full

        self.gray_buffer.append(prev_gray)
        self.light_trail.add_frame(prev_gray)

        frame_idx = 1

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray_full = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray_full = cv2.GaussianBlur(curr_gray_full, (3, 3), 0)

            if self.proc_scale != 1.0:
                curr_gray = cv2.resize(curr_gray_full, (0, 0), fx=self.proc_scale, fy=self.proc_scale)
                scale_up = self.proc_scale
            else:
                curr_gray = curr_gray_full
                scale_up = 1.0

            self.gray_buffer.append(curr_gray)
            self.light_trail.add_frame(curr_gray)

            # 生成光軌效果
            if self.light_trail_mode == 'weighted':
                trail_gray = self.light_trail.get_light_trail()
            elif self.light_trail_mode == 'max':
                trail_gray = self.light_trail.get_max_projection()
            else:
                trail_gray = self.light_trail.get_enhanced_trail()

            # Multi-frame 差分
            N = max(1, int(round(self.target_window_seconds * self.fps)))
            if len(self.gray_buffer) >= N:
                prev_gray_k = self.gray_buffer[-N]
            else:
                prev_gray_k = self.gray_buffer[0]

            delta_standard = cv2.absdiff(curr_gray, prev_gray_k)
            delta_standard = cv2.medianBlur(delta_standard, 3)

            # 光軌差分
            if trail_gray is not None and len(self.gray_buffer) >= self.light_trail.trail_length:
                delta_trail = cv2.absdiff(curr_gray, trail_gray)
                delta_trail = cv2.medianBlur(delta_trail, 3)
                delta = cv2.max(delta_standard, delta_trail)
            else:
                delta = delta_standard
            
            # 保留 prev_gray_k 供場景檢測使用
            prev_gray_for_scene = prev_gray_k

            # 場景切換檢查（更嚴格 + 多重驗證）
            scene_debug = {}
            is_scene_change, reasons = self.is_scene_change(
                delta_standard, curr_gray, prev_gray_for_scene, 
                threshold_ratio=0.05, debug_info=scene_debug
            )
            
            if is_scene_change:
                self.scene_change_frames.append(frame_idx)
                self.scene_change_buffer.append(frame_idx)
                
                if self.debug:
                    self.log(f"  [場景切換] Frame {frame_idx} - 原因: {', '.join(reasons)}")
                    self.log(f"    詳細: change_ratio={scene_debug.get('change_ratio', 0):.3f}, "
                            f"hist_diff={scene_debug.get('hist_diff', 0):.3f}, "
                            f"brightness_change={scene_debug.get('brightness_change', 0):.1f}")
                
                # 清空所有追蹤狀態
                self.active_tracks.clear()
                self.light_trail.frame_buffer.clear()
                self.gray_buffer.clear()
                self.star_background.position_history.clear()
                self.pending_spots.clear()
                
                # 重新放入當前幀
                self.gray_buffer.append(curr_gray)
                prev_gray = curr_gray
                frame_idx += 1
                continue
            
            # **場景切換冷卻期：直接跳過偵測**
            if self.is_in_scene_cooldown(frame_idx):
                if self.debug and frame_idx % 25 == 0:
                    self.log(f"  [冷卻期] Frame {frame_idx} - 跳過偵測")
                frame_idx += 1
                continue

            # 動態閾值
            dynamic_threshold = max(self.base_threshold, int(self.base_threshold * np.sqrt(max(1, N))))
            spots = self.find_bright_spots(delta, threshold=dynamic_threshold, scale_up_factor=scale_up)

            # **更新恆星背景模型**
            self.star_background.update(spots, frame_idx)

            # **過濾靜止恆星**
            filtered_spots = []
            for spot in spots:
                if not self.star_background.is_static_star(spot, frame_idx, min_appearances=15):
                    filtered_spots.append(spot)
            
            if self.debug and len(spots) != len(filtered_spots):
                self.log(f"  [恆星過濾] Frame {frame_idx}: {len(spots)} -> {len(filtered_spots)} spots")

            spots = filtered_spots

            # pending_spots 用於穩定性檢查
            self.pending_spots.append([(s['x'], s['y']) for s in spots])

            # 單幀保護
            if len(spots) > self.spot_frame_cap:
                if self.debug:
                    self.log(f"[跳過] Frame {frame_idx} - spots {len(spots)} 太多")
                frame_idx += 1
                continue

            if self.debug and (len(spots) > 0 or len(self.active_tracks) > 0):
                self.save_debug_frame(curr_frame, 
                                    cv2.resize(delta_standard, (self.width, self.height)) if self.proc_scale != 1.0 else delta_standard,
                                    cv2.resize(delta, (self.width, self.height)) if self.proc_scale != 1.0 else delta,
                                    spots, frame_idx)

            # 更新軌跡遺失計數
            for track in self.active_tracks:
                track['lost_frames'] += 1

            # 匹配點到軌跡
            matched_tracks = set()
            for spot in spots:
                if not self.stable_spot_exists(spot, radius=6, required_frames=2):
                    continue

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
                            f"幀 {meteor['frame_range'][0]}-{meteor['frame_range'][1]}, "
                            f"長度 {meteor['length']:.1f}px, "
                            f"品質 {meteor['quality_score']:.1f}")

            # 定期清理恆星記錄
            if frame_idx % 60 == 0:
                self.star_background.cleanup_old_entries(frame_idx)

            frame_idx += 1
            if frame_idx % 100 == 0:
                self.log(f"處理進度: {frame_idx}/{self.total_frames}, Active: {len(self.active_tracks)}, Meteors: {len(self.meteors)}")

        # 處理剩餘軌跡
        for track in self.active_tracks:
            meteor = self.finalize_track(track)
            if meteor is not None:
                self.meteors.append(meteor)

        cap.release()

        self.log(f"\n總共偵測到 {len(self.meteors)} 顆流星")
        self.log(f"拒絕的軌跡: {len(self.rejected_tracks)} 條")
        self.log(f"場景切換次數: {len(self.scene_change_frames)}")

        # 儲存被拒絕的軌跡分析
        if self.debug and len(self.rejected_tracks) > 0:
            self.save_rejection_analysis()

        self.save_results()
        if len(self.meteors) > 0:
            self.create_annotated_video()
        else:
            self.log("⚠️ 未偵測到流星")

        return self.meteors

    def save_rejection_analysis(self):
        """儲存被拒絕軌跡的分析報告"""
        rejection_stats = defaultdict(int)
        for track in self.rejected_tracks:
            reason = track.get('reject_reason', 'unknown')
            rejection_stats[reason] += 1

        analysis = {
            'total_rejected': len(self.rejected_tracks),
            'rejection_reasons': dict(rejection_stats),
            'sample_tracks': []
        }

        # 取每種拒絕原因的樣本
        reason_samples = defaultdict(list)
        for track in self.rejected_tracks:
            reason = track.get('reject_reason', 'unknown')
            if len(reason_samples[reason]) < 3:
                reason_samples[reason].append({
                    'track_id': track['track_id'],
                    'start_frame': track['start_frame'],
                    'duration': len(track['points']),
                    'quality_score': track.get('quality_score', 0)
                })

        analysis['samples_by_reason'] = dict(reason_samples)

        with open(self.output_folder / "rejection_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        self.log(f"✓ 拒絕分析已儲存: rejection_analysis.json")
        self.log(f"  主要拒絕原因:")
        for reason, count in sorted(rejection_stats.items(), key=lambda x: -x[1])[:5]:
            self.log(f"    - {reason}: {count} 次")

    def save_results(self):
        result = {
            'video': str(self.video_path),
            'total_meteors': len(self.meteors),
            'fps': self.fps,
            'sensitivity': self.sensitivity,
            'quality_threshold': self.quality_threshold,
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

    def create_annotated_video(self):
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

            if frame_idx in self.scene_change_frames:
                cv2.putText(frame, "SCENE CHANGE", (self.width//2-100, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            for meteor in self.meteors:
                start_f, end_f = meteor['frame_range']
                if start_f <= frame_idx <= end_f:
                    points = meteor['trajectory']
                    for i in range(1, len(points)):
                        p1 = (points[i-1]['x'], points[i-1]['y'])
                        p2 = (points[i]['x'], points[i]['y'])
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)

                    for point in points:
                        if point['frame'] == frame_idx:
                            cv2.circle(frame, (point['x'], point['y']),
                                       8, (0, 255, 255), 2)
                            cv2.circle(frame, (point['x'], point['y']),
                                       3, (0, 0, 255), -1)

                    bbox = meteor['bbox']
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                  (255, 0, 0), 1)

                    label = f"{meteor['meteor_id']} | Q:{meteor['quality_score']:.0f}"
                    cv2.putText(frame, label,
                                (bbox[0], max(15, bbox[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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
        description='進階流星偵測系統 - 多層級 + 恆星過濾 + 品質評分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 高敏感度（容易偵測但誤報多）
  python meteor_advanced.py --video me.mp4 --sensitivity high --debug

  # 中等敏感度（平衡）
  python meteor_advanced.py --video me.mp4 --sensitivity medium

  # 低敏感度（只抓明顯流星）
  python meteor_advanced.py --video me.mp4 --sensitivity low

  # 自訂參數
  python meteor_advanced.py --video me.mp4 --trail-length 5 --proc-scale 0.7 --debug
        """
    )
    parser.add_argument('--video', type=str, required=True, help='影片路徑')
    parser.add_argument('--sensitivity', type=str, default='medium',
                       choices=['low', 'medium', 'high'],
                       help='敏感度等級 (low=只抓明顯流星, medium=平衡, high=容易偵測)')
    parser.add_argument('--trail-length', type=int, default=3,
                       help='光軌累積幀數 (建議 3-8，預設 3)')
    parser.add_argument('--trail-mode', type=str, default='enhanced',
                       choices=['weighted', 'max', 'enhanced'],
                       help='光軌模式 (預設 enhanced)')
    parser.add_argument('--proc-scale', type=float, default=1.0,
                       help='處理解析度縮放 (0.5 ~ 1.0)，0.5 可加速並降低噪音')
    parser.add_argument('--debug', action='store_true',
                       help='啟用調試模式（輸出偵測過程圖片 + 拒絕分析）')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ 找不到影片: {args.video}")
        return

    output_folder = Path("output_advanced") / Path(args.video).stem
    tracker = AdvancedMeteorTracker(
        args.video,
        output_folder,
        debug=args.debug,
        light_trail_length=args.trail_length,
        light_trail_mode=args.trail_mode,
        proc_scale=args.proc_scale,
        sensitivity=args.sensitivity
    )
    tracker.process_video()


if __name__ == '__main__':
    main()