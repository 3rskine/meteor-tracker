
"""
光軌增強版流星偵測系統 - 進階優化版（含 OSD 過濾）
新增功能：
1. 多層級偵測（同時用不同參數偵測）
2. 恆星過濾器（背景減除 + 靜止點過濾）
3. 場景切換保護增強
4. 軌跡品質評分系統
5. 自適應參數調整
6. OSD（影像下方顯示的跳動數字）自動偵測與遮罩
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
import sys

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
    """光軌 (Light Trail) 處理器 - 累積移動物體軌跡（含低FPS補線功能）"""

    def __init__(self, trail_length=5, decay_type='uniform', enable_interpolation=True):
        self.trail_length = trail_length
        self.decay_type = decay_type
        self.frame_buffer = deque(maxlen=trail_length)
        self.weights = self._compute_weights()
        self.enable_interpolation = enable_interpolation
        self.prev_frame = None   # 用來記住上一個 frame

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

    # ------------------------ 這裡是你最需要改的地方 ------------------------
    def add_frame(self, frame):
        frame_copy = frame.copy()

        # 如果啟用補線 & 有上一張 frame，就補間軌跡
        if self.enable_interpolation and self.prev_frame is not None:
            self._interpolate_trail(self.prev_frame, frame_copy)

        self.frame_buffer.append(frame_copy)
        self.prev_frame = frame.copy()
    # -----------------------------------------------------------------------

    def _interpolate_trail(self, prev_frame, curr_frame):
        """在兩個 frame 間補線，讓低FPS時軌跡仍連續"""

        # 找出亮點（閾值可調，越低越敏感）
        threshold = 200
        p1_list = np.argwhere(prev_frame > threshold)
        p2_list = np.argwhere(curr_frame > threshold)

        # 沒亮點直接跳過
        if len(p1_list) == 0 or len(p2_list) == 0:
            return

        # 計算兩張圖亮點的平均位置（當作軌跡位置）
        p1 = p1_list.mean(axis=0).astype(int)
        p2 = p2_list.mean(axis=0).astype(int)

        # cv2 的座標要反過來 (x,y)
        cv2.line(curr_frame,
                 (p1[1], p1[0]),
                 (p2[1], p2[0]),
                 color=255,
                 thickness=1)

    # -----------------------------------------------------------------------

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

        # ============ OSD 探測相關 ============
        self.osd_mask = None
        self._osd_change_count = None
        self.osd_warmup_frames = min(200, max(30, int(self.fps * 5)))
        self.osd_min_freq_ratio = 0.12
        self.osd_dilate = 5
        self.manual_osd_bbox = (0, 460, 480, 80) #default mask

        # **新增：如果有手動指定，立即建立 mask**
        self._manual_osd_mask = None  # 先準備變數

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
            self.quality_threshold = 70  # 較高的品質門檻
            self.base_threshold = 6
            self.min_track_duration = 3  # 最少持續 3 幀
            
        elif level == 'low':
            # 低敏感度：只抓明顯流星
            self.min_movement_pixels = 3
            self.movement_check_frames = 3
            self.lost_track_threshold = 8
            self.min_area_pixels = 5
            self.spot_frame_cap = 150
            self.min_max_brightness = 0.2
            self.min_speed_px_per_frame = 2.0
            self.quality_threshold = 30  
            self.base_threshold = 12
            self.min_track_duration = 4 
            
        else:  # medium (default)
            self.min_movement_pixels = 10
            self.movement_check_frames = 1
            self.lost_track_threshold = 10
            self.min_area_pixels = 3 ##less impor
            self.spot_frame_cap = 200
            self.min_max_brightness = 0.5
            self.min_speed_px_per_frame = 0.1
            self.quality_threshold = 10  
            self.base_threshold = 4 ##important
            self.min_track_duration = 3  

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

        # **新增：顯示 OSD 遮罩區域（半透明紅色）**
        if self.osd_mask is not None:
            # 將 scaled mask 放大回原解析度
            mask_full = cv2.resize(self.osd_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            # 創建紅色半透明覆蓋層
            overlay = vis.copy()
            overlay[mask_full > 0] = [0, 0, 255]  # 紅色
            # 混合（透明度 30%）
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            
            # 在遮罩邊界畫框
            contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

        # 標記 spots（綠色=移動，紅色=靜止恆星，黃色=在OSD內被過濾）
        for spot in spots:
            if self.star_background.is_static_star(spot, frame_idx):
                cv2.circle(vis, (spot['x'], spot['y']), 5, (0, 0, 255), 2)  # 紅色（恆星）
            else:
                cv2.circle(vis, (spot['x'], spot['y']), 5, (0, 255, 0), 2)  # 綠色（移動）

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
        
        # **新增：OSD 資訊顯示**
        osd_info = f"OSD Mask: {'Active' if self.osd_mask is not None else 'None'}"
        info = (f"Frame {frame_idx} | Spots: {len(spots)} | "
            f"Active: {len(self.active_tracks)} | {osd_info}")
        cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

        filename = self.debug_folder / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(filename), combined)

    def _spot_in_osd(self, spot):
        """spot 的座標是原解析度 (x,y)；我們將其轉到 scaled space 檢查 osd_mask"""
        if self.osd_mask is None:
            return False
        # map to scaled coordinates
        sx = int(round(spot['x'] * self.proc_scale))
        sy = int(round(spot['y'] * self.proc_scale))
        h, w = self.osd_mask.shape
        if sx < 0 or sy < 0 or sx >= w or sy >= h:
            return False
        return bool(self.osd_mask[sy, sx])

def process_input(self):
    """統一的輸入處理（影片/圖片/圖片序列）"""
    
    self.log("=" * 80)
    self.log(f"進階流星偵測系統 - {self.input_type} 模式")
    self.log("=" * 80)
    
    if self.input_type == 'video':
        return self._process_video()
    elif self.input_type == 'image':
        return self._process_single_image()
    elif self.input_type == 'image_sequence':
        return self._process_image_sequence()


    def _process_single_image(self):
        """處理單張圖片（長曝光流星軌跡）"""
        
        self.log(f"處理圖片: {self.input_path}")
        
        # 讀取圖片
        image = cv2.imread(str(self.input_path))
        if image is None:
            self.log("❌ 無法讀取圖片")
            return []
        
        self.height, self.width = image.shape[:2]
        self.log(f"圖片尺寸: {self.width}×{self.height}")
        
        # 轉灰階
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # **關鍵：單張圖片使用不同偵測策略**
        # 不用幀差，直接找亮線
        meteors = self._detect_trails_in_static_image(gray, image)
        
        self.log(f"偵測到 {len(meteors)} 條軌跡")
        
        # 儲存結果
        self.save_results()
        
        if len(meteors) > 0:
            self._create_annotated_image(image)
        
        return meteors


        def _detect_trails_in_static_image(self, gray, original):
            """在靜態圖片中偵測流星軌跡"""
            
            # 方法 1：Hough Line Transform（論文方法）
            edges = cv2.Canny(gray, 50, 150)
            
            # 偵測直線
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,      # 最少投票數
                minLineLength=30,  # 最短線段
                maxLineGap=10      # 最大間隔
            )
            
            if lines is None:
                return []
            
            meteors = []
            
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                
                # 計算線段特性
                length = np.hypot(x2 - x1, y2 - y1)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # 過濾條件
                if length < 20:  # 太短
                    continue
                
                # 檢查亮度
                brightness = self._check_line_brightness(gray, x1, y1, x2, y2)
                
                if brightness < 50:  # 太暗
                    continue
                
                # 建立軌跡資料
                meteor = {
                    'meteor_id': f"static_meteor_{i:03d}",
                    'trajectory': [
                        {'x': x1, 'y': y1, 'frame': 0, 'brightness': brightness},
                        {'x': x2, 'y': y2, 'frame': 0, 'brightness': brightness}
                    ],
                    'length': float(length),
                    'angle': float(angle),
                    'max_brightness': float(brightness),
                    'bbox': [min(x1, x2)-10, min(y1, y2)-10, 
                            abs(x2-x1)+20, abs(y2-y1)+20]
                }
                
                meteors.append(meteor)
            
            return meteors


            def _check_line_brightness(self, gray, x1, y1, x2, y2):
                """檢查線段的平均亮度"""
                
                # 沿著線段採樣
                num_samples = int(np.hypot(x2 - x1, y2 - y1))
                if num_samples < 2:
                    return 0
                
                xs = np.linspace(x1, x2, num_samples).astype(int)
                ys = np.linspace(y1, y2, num_samples).astype(int)
                
                # 邊界檢查
                h, w = gray.shape
                valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                
                if not valid.any():
                    return 0
                
                # 計算平均亮度
                brightnesses = gray[ys[valid], xs[valid]]
                return np.mean(brightnesses)


            def _process_image_sequence(self):
                """處理圖片序列（類似影片處理）"""
                
                self.log(f"處理圖片序列: {len(self.image_files)} 張圖片")
                
                if len(self.image_files) == 0:
                    self.log("❌ 找不到圖片")
                    return []
                
                # 讀取第一張確定尺寸
                first_image = cv2.imread(str(self.image_files[0]))
                if first_image is None:
                    self.log("❌ 無法讀取第一張圖片")
                    return []
                
                self.height, self.width = first_image.shape[:2]
                
                # 初始化（類似影片處理）
                prev_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.GaussianBlur(prev_gray, (3, 3), 0)
                
                if self.proc_scale != 1.0:
                    prev_gray = cv2.resize(prev_gray, (0, 0), 
                                        fx=self.proc_scale, fy=self.proc_scale)
                
                self.gray_buffer.append(prev_gray)
                self.light_trail.add_frame(prev_gray)
                
                # **處理每張圖片（類似影片的幀處理）**
                for frame_idx, image_path in enumerate(self.image_files[1:], start=1):
                    
                    # 讀取圖片
                    curr_frame = cv2.imread(str(image_path))
                    if curr_frame is None:
                        self.log(f"⚠️ 無法讀取: {image_path}")
                        continue
                    
                    # 轉灰階
                    curr_gray_full = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray_full = cv2.GaussianBlur(curr_gray_full, (3, 3), 0)
                    
                    # **以下完全使用原本的影片處理邏輯**
                    if self.proc_scale != 1.0:
                        curr_gray = cv2.resize(curr_gray_full, (0, 0),
                                            fx=self.proc_scale, fy=self.proc_scale)
                    else:
                        curr_gray = curr_gray_full
                    
                    self.gray_buffer.append(curr_gray)
                    self.light_trail.add_frame(curr_gray)
                    
                    # ... 原本的偵測邏輯 ...
                    # （完全相同，不需修改）
                    
                    if frame_idx % 10 == 0:
                        self.log(f"處理進度: {frame_idx}/{len(self.image_files)}")
                
                # 處理剩餘軌跡
                for track in self.active_tracks:
                    meteor = self.finalize_track(track)
                    if meteor is not None:
                        self.meteors.append(meteor)
                
                self.log(f"總共偵測到 {len(self.meteors)} 顆流星")
                self.save_results()
                
                return self.meteors


            def _create_annotated_image(self, image):
                """為單張圖片建立標記版本"""
                
                self.log("生成標記圖片...")
                
                output_image_path = self.output_folder / f"{self.input_path.stem}_annotated.jpg"
                
                annotated = image.copy()
                
                for meteor in self.meteors:
                    points = meteor['trajectory']
                    
                    # 畫軌跡線
                    for i in range(1, len(points)):
                        p1 = (points[i-1]['x'], points[i-1]['y'])
                        p2 = (points[i]['x'], points[i]['y'])
                        cv2.line(annotated, p1, p2, (0, 255, 0), 2)
                    
                    # 畫 bounding box
                    bbox = meteor['bbox']
                    cv2.rectangle(annotated,
                                (bbox[0], bbox[1]),
                                (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                (255, 0, 0), 2)
                    
                    # 標記 ID
                    label = meteor['meteor_id']
                    cv2.putText(annotated, label,
                            (bbox[0], bbox[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # 加總結資訊
                info = f"Meteors: {len(self.meteors)}"
                cv2.putText(annotated, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                cv2.imwrite(str(output_image_path), annotated)
                self.log(f"✓ 標記圖片已儲存: {output_image_path}")

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


# ==================== 批次處理器（支援圖片和影片） ====================

class BatchMeteorProcessor:
    """批次處理資料夾內的所有圖片"""
    
    def __init__(self, input_folder, output_base_folder, **kwargs):
        self.input_folder = Path(input_folder)
        self.output_base_folder = Path(output_base_folder)
        self.output_base_folder.mkdir(parents=True, exist_ok=True)
        
        self.batch_log_file = self.output_base_folder / "batch_processing.log"
        self.results_summary = []
        
        # 自動套用 OSD 遮罩
        self.default_osd_bbox = (0, 460, 470, 80)
        self.debug = kwargs.get('debug', False)
        
        with open(self.batch_log_file, 'w', encoding='utf-8') as f:
            f.write(f"批次處理開始: {datetime.now()}\n")
            f.write(f"OSD 遮罩: {self.default_osd_bbox}\n")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.batch_log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def find_files(self):
        files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            files.extend(self.input_folder.glob(ext))
        return sorted(set(files))
    
    def _check_line_brightness(self, gray, x1, y1, x2, y2):
        """檢查線段平均亮度"""
        num_samples = int(np.hypot(x2 - x1, y2 - y1))
        if num_samples < 2:
            return 0
        
        xs = np.linspace(x1, x2, num_samples).astype(int)
        ys = np.linspace(y1, y2, num_samples).astype(int)
        
        h, w = gray.shape
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        
        if not valid.any():
            return 0
        
        return np.mean(gray[ys[valid], xs[valid]])
    
    def process_single_image(self, image_path, image_index, total_files):
        """處理單張圖片"""
        self.log("=" * 80)
        self.log(f"處理圖片 [{image_index}/{total_files}]: {image_path.name}")
        self.log("=" * 80)
        
        output_folder = self.output_base_folder / image_path.stem
        output_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            start_time = datetime.now()
            
            # 讀取圖片
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"無法讀取圖片")
            
            height, width = image.shape[:2]
            self.log(f"圖片尺寸: {width}×{height}")
            
            # 轉灰階
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # === 套用 OSD 遮罩 ===
            osd_mask = np.ones_like(gray, dtype=np.uint8) * 255
            x, y, w, h = self.default_osd_bbox
            
            # 確保不超出邊界
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            x_end = min(x + w, width)
            y_end = min(y + h, height)
            
            osd_mask[y:y_end, x:x_end] = 0
            osd_region = (x, y, w, h)
            
            # 套用遮罩
            gray_masked = cv2.bitwise_and(gray_blurred, gray_blurred, mask=osd_mask)
            
            # Hough Line Transform
            edges = cv2.Canny(gray_masked, 30, 100, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=5)
            
            meteors = []
            
            if lines is not None:
                self.log(f"偵測到 {len(lines)} 條線段")
                
                for i, line in enumerate(lines):
                    x1, y1, x2, y2 = line[0]
                    
                    # 檢查是否在 OSD 區域
                    mid_y = (y1 + y2) // 2
                    if mid_y >= y and mid_y <= y + h:
                        continue
                    
                    length = np.hypot(x2 - x1, y2 - y1)
                    if length < 20:
                        continue
                    
                    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                    if angle < 5:  # 排除水平線
                        continue
                    
                    brightness = self._check_line_brightness(gray, x1, y1, x2, y2)
                    if brightness < 80:
                        continue
                    
                    meteor = {
                        'meteor_id': f"meteor_{len(meteors):03d}",
                        'start_point': [int(x1), int(y1)],
                        'end_point': [int(x2), int(y2)],
                        'length': float(length),
                        'angle': float(angle),
                        'brightness': float(brightness),
                        'bbox': [int(min(x1, x2)-10), int(min(y1, y2)-10), 
                                int(abs(x2-x1)+20), int(abs(y2-y1)+20)]
                    }
                    meteors.append(meteor)
                
                self.log(f"通過過濾: {len(meteors)} 條軌跡")
            else:
                self.log("未偵測到線段")
            
            # 儲存結果圖片
            if meteors:
                result_img = image.copy()
                
                for meteor in meteors:
                    x1, y1 = meteor['start_point']
                    x2, y2 = meteor['end_point']
                    
                    cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(result_img, (x1, y1), 6, (0, 0, 255), -1)
                    cv2.circle(result_img, (x2, y2), 6, (255, 0, 0), -1)
                    
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    cv2.putText(result_img, meteor['meteor_id'],
                              (mid_x, mid_y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # 畫 OSD 區域
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(result_img, "OSD MASKED", (x+10, y+20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 總結資訊
                cv2.putText(result_img, f"Meteors: {len(meteors)}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                output_image = output_folder / f"{image_path.stem}_detected.jpg"
                cv2.imwrite(str(output_image), result_img)
                self.log(f"✓ 標記圖片: {output_image}")
                
                if self.debug:
                    cv2.imwrite(str(output_folder / f"{image_path.stem}_edges.jpg"), edges)
            
            # 儲存 JSON
            result_json = {
                'image': str(image_path),
                'size': [width, height],
                'osd_region': list(osd_region),
                'meteor_count': len(meteors),
                'meteors': meteors
            }
            
            with open(output_folder / "results.json", 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                'file_name': image_path.name,
                'file_path': str(image_path),
                'meteor_count': len(meteors),
                'processing_time_seconds': processing_time,
                'status': 'success'
            }
            
            self.log(f"✓ 完成: {len(meteors)} 條軌跡，耗時 {processing_time:.2f}s\n")
            
        except Exception as e:
            self.log(f"❌ 錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            
            result = {
                'file_name': image_path.name,
                'file_path': str(image_path),
                'status': 'failed',
                'error': str(e)
            }
        
        self.results_summary.append(result)
        return result
    
    def process_all(self):
        """處理所有檔案"""
        self.log("開始批次處理流星偵測")
        self.log(f"輸入資料夾: {self.input_folder}")
        self.log(f"輸出資料夾: {self.output_base_folder}")
        self.log(f"OSD 遮罩: {self.default_osd_bbox}\n")
        
        files = self.find_files()
        
        if not files:
            self.log("❌ 找不到任何圖片檔案")
            return []
        
        self.log(f"找到 {len(files)} 張圖片\n")
        
        batch_start = datetime.now()
        
        # 處理每張圖片
        for i, image_path in enumerate(files, 1):
            self.process_single_image(image_path, i, len(files))
        
        batch_end = datetime.now()
        total_time = (batch_end - batch_start).total_seconds()
        
        # 生成總結報告
        self.generate_summary_report(total_time)
        
        return self.results_summary
    
    def generate_summary_report(self, total_time):
        """生成批次處理總結報告"""
        self.log("\n" + "=" * 80)
        self.log("批次處理完成")
        self.log("=" * 80)
        
        success_count = sum(1 for r in self.results_summary if r['status'] == 'success')
        failed_count = len(self.results_summary) - success_count
        total_meteors = sum(r.get('meteor_count', 0) for r in self.results_summary)
        
        self.log(f"總檔案數: {len(self.results_summary)}")
        self.log(f"成功: {success_count}, 失敗: {failed_count}")
        self.log(f"總偵測流星數: {total_meteors}")
        self.log(f"總耗時: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
        
        if success_count > 0:
            avg_time = total_time / success_count
            self.log(f"平均每張圖片: {avg_time:.2f} 秒")
        
        # 儲存 JSON 報告
        summary_file = self.output_base_folder / "batch_summary.json"
        summary_data = {
            'total_files': len(self.results_summary),
            'success_count': success_count,
            'failed_count': failed_count,
            'total_meteors': total_meteors,
            'total_time_seconds': total_time,
            'osd_bbox': self.default_osd_bbox,
            'results': self.results_summary
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"\n✓ 總結報告已儲存: {summary_file}")
        
        # 顯示每個檔案的結果（前20個）
        self.log("\n詳細結果（前20個）:")
        for i, result in enumerate(self.results_summary[:20], 1):
            if result['status'] == 'success':
                self.log(f"  {i}. {result['file_name']}: "
                        f"{result['meteor_count']} 條軌跡 "
                        f"({result['processing_time_seconds']:.2f}s)")
            else:
                self.log(f"  {i}. {result['file_name']}: 失敗 - {result.get('error', 'unknown')}")
        
        if len(self.results_summary) > 20:
            self.log(f"  ... 還有 {len(self.results_summary) - 20} 個檔案")


def parse_osd_bbox(arg):
    """解析 'x,y,w,h' 字串為 tuple"""
    try:
        parts = [int(p) for p in arg.split(',')]
        if len(parts) != 4:
            raise ValueError()
        return tuple(parts)
    except Exception:
        raise argparse.ArgumentTypeError("OSD bbox 格式錯誤，應為 x,y,w,h（整數）")


def main():
    parser = argparse.ArgumentParser(
        description='流星偵測系統 - 批次處理圖片（自動套用 OSD 遮罩）'
    )
    
    parser.add_argument('--folder', type=str, required=True,
                       help='批次處理資料夾')
    
    parser.add_argument('--debug', action='store_true',
                       help='啟用 debug 模式（儲存邊緣檢測圖）')
    
    parser.add_argument('--osd-bbox', type=parse_osd_bbox, default=None,
                       help='覆蓋預設 OSD 區域（格式: x,y,w,h）。預設: 0,460,470,80')
    
    args = parser.parse_args()
    
    print(f"🔄 批次處理模式: {args.folder}")
    print(f"✓ 自動套用 OSD 遮罩: (0, 460, 470, 80)\n")
    
    processor = BatchMeteorProcessor(
        input_folder=args.folder,
        output_base_folder=Path("output_batch"),
        debug=args.debug
    )
    
    # 如果有手動指定，覆蓋預設值
    if args.osd_bbox:
        processor.default_osd_bbox = args.osd_bbox
        print(f"✓ 覆蓋為手動指定: {args.osd_bbox}\n")
    
    processor.process_all()


if __name__ == '__main__':
    main()