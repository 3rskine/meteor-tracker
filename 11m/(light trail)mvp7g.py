#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光鬼增強版流星偵測系統 - Light Trail + Kalman + 合併
已整合：對亮背景友好的參數與保護機制（morphology、min_area、stable_spot、spot-cap、proc-scale）
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

    def __init__(self, trail_length=10, decay_type='uniform'):
        """
        Args:
            trail_length: 累積幀數 (建議 3-8，預設 3)
            decay_type: 衰減模式 ('exponential', 'linear', 'uniform')
        """
        self.trail_length = trail_length
        self.decay_type = decay_type
        self.frame_buffer = deque(maxlen=trail_length)
        self.weights = self._compute_weights()

    def _compute_weights(self):
        """計算時間衰減權重"""
        n = max(1, self.trail_length)
        if self.decay_type == 'exponential':
            # 指數衰減：最新幀權重最高（調整衰減速率）
            weights = np.exp(-np.arange(n) * 0.35)
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
        """加入新幀到緩衝區（frame 應為灰階 uint8）"""
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

        return np.clip(result, 0, 255).astype(np.uint8)

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
                 light_trail_length=3, light_trail_mode='enhanced', proc_scale=1.0):
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

        # === 光鬼處理器 ===
        self.light_trail = LightTrailProcessor(
            trail_length=light_trail_length,
            decay_type='linear'
        )
        self.light_trail_mode = light_trail_mode  # 'weighted', 'max', 'enhanced'

        # 偵測結果
        self.meteors = []
        self.log_file = self.output_folder / "detection.log"
        self.active_tracks = []
        self.meteor_id_counter = 0
        self.scene_change_frames = []

        # === 優化參數（針對亮背景 / 環境噪音）===
        self.min_movement_pixels = 2
        self.movement_check_frames = 2
        self.lost_track_threshold = 10

        # Multi-frame 差分設定
        self.target_window_seconds = 0.15
        self.gray_buffer = deque(maxlen=max(60, int(self.fps * 2)))

        # 近期完成軌跡緩衝 & pending spots（穩定性檢查）
        self.recent_finalized = deque(maxlen=120)
        self.pending_spots = deque(maxlen=8)  # 儲存最近幾幀 spots（每幀 list of (x,y)）

        # morphology / area thresholds (亮背景友好)
        self.morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area_pixels = 3  # 最小面積門檻，丟掉 <8 px 的碎片

        # spots safety
        self.spot_frame_cap = 200  # 單幀 spots 超過就跳過那幀

        # is_meteor thresholds
        self.min_max_brightness = 1.5
        self.min_speed_px_per_frame = 1.0

        # Debug 設定
        if debug:
            self.debug_folder = self.output_folder / "debug_frames"
            self.debug_folder.mkdir(exist_ok=True)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Log start: {datetime.now()}\n")
            f.write(f"光鬼設定: length={light_trail_length}, mode={light_trail_mode}, proc_scale={self.proc_scale}\n")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

    # ==================== 場景檢測（光鬼版本需要更寬鬆）====================
    def is_scene_change(self, delta, threshold_ratio=0.08, debug_info=None):
        total_pixels = delta.shape[0] * delta.shape[1]
        changed_pixels = np.count_nonzero(delta > 20)
        change_ratio = changed_pixels / float(total_pixels)

        if debug_info is not None and self.debug:
            debug_info['change_ratio'] = change_ratio
            debug_info['threshold'] = threshold_ratio

        return change_ratio > threshold_ratio

    # ==================== 亮點偵測（使用光鬼）====================
    def find_bright_spots(self, delta_frame, threshold=2, scale_up_factor=1.5):
        """
        delta_frame: 處理用的灰階差分 (可能為縮小版)
        threshold: 二值化門檻 (uint)
        scale_up_factor: 若 delta_frame 是縮小的，回傳的座標會放大回原尺寸
        """
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

                    # scale up coordinates to original frame if needed
                    if scale_up_factor != 1.0:
                        inv = 1.0 / scale_up_factor
                        cx = int(round(cx * inv))
                        cy = int(round(cy * inv))
                        x = int(round(x * inv))
                        y = int(round(y * inv))
                        w = int(round(w * inv))
                        h = int(round(h * inv))
                        # brightness stays relative to small frame; it's ok as heuristic

                    spots.append({
                        'x': cx, 'y': cy, 'area': area,
                        'brightness': float(brightness), 'contour': contour,
                        'bbox': (x, y, w, h)
                    })
        return spots

    # ==================== 穩定性判定 =====================
    def stable_spot_exists(self, spot, radius=6, required_frames=2):
        """檢查 spot 是否在最近 required_frames 幀內穩定出現（使用原始座標）"""
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

    # ==================== 軌跡一致性檢查 =====================
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
                if self.debug:
                    self.log(f"    [預測偏差] Track {track['track_id']}: {pred_distance:.1f} > {max_deviation:.1f}")
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
                if self.debug:
                    self.log(f"    [方向變化] Track {track['track_id']}: {angle_diff:.1f}°")
                return False

        return True

    # ==================== 移動檢查（寬鬆版）====================
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
            if self.debug:
                self.log(f"    [靜止點] Track {track['track_id']}: {check_window}幀僅移動 {straight_distance:.1f}px")
            return False

        average_speed = straight_distance / check_window
        if average_speed < 2.5:
            if self.debug:
                self.log(f"    [速度過慢] Track {track['track_id']}: {average_speed:.2f}px/frame")
            return False

        return True

    # ==================== 軌跡匹配 =====================
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

    # ==================== 軌跡驗證（極寬鬆版，專注短軌跡）====================
    def is_meteor_track(self, track):
        points = track['points']
        if len(points) < 2:
            return False

        start_point = points[0]
        end_point = points[-1]
        straight_distance = np.hypot(end_point['x'] - start_point['x'],
                                    end_point['y'] - start_point['y'])
        if straight_distance < 2.0:
            if self.debug:
                self.log(f"  [靜止點] Track {track['track_id']}: 僅移動 {straight_distance:.1f}px")
            return False

        start_frame = track['start_frame']
        for scene_frame in self.scene_change_frames:
            if abs(start_frame - scene_frame) <= 3:
                return False

        total_length = self.calculate_trajectory_length(points)
        if total_length < 3:
            if self.debug:
                self.log(f"  [太短] Track {track['track_id']}: {total_length:.1f}px")
            return False

        duration = points[-1]['frame'] - points[0]['frame']
        if duration == 0:
            duration = 1
        speed = total_length / duration

        if speed < self.min_speed_px_per_frame or speed > 1000:
            if self.debug:
                self.log(f"  [速度異常] Track {track['track_id']}: {speed:.1f}px/frame")
            return False

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

                if mean_deviation > 50.0:
                    if self.debug:
                        self.log(f"  [非線性] Track {track['track_id']}: {mean_deviation:.1f}")
                    return False
            except Exception:
                pass

        brightnesses = [p.get('brightness', 0.0) for p in points]
        max_brightness = max(brightnesses) if brightnesses else 0.0
        if max_brightness < self.min_max_brightness:
            if self.debug:
                self.log(f"  [亮度低] Track {track['track_id']}: {max_brightness:.1f}")
            return False

        return True

    # ==================== 軌跡操作 =====================
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

    # ==================== 合併/計算輔助函式 =====================
    def should_merge_meteors(self, meteor1, meteor2):
        r1 = meteor1['frame_range']
        r2 = meteor2['frame_range']
        time_gap = max(0, r2[0] - r1[1], r1[0] - r2[1])

        if time_gap > 30:
            return False

        angle1 = meteor1['angle']
        angle2 = meteor2['angle']
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        if angle_diff > 30:
            return False

        traj1 = meteor1['trajectory']
        traj2 = meteor2['trajectory']
        distances = [
            np.hypot(traj1[-1]['x'] - traj2[0]['x'], traj1[-1]['y'] - traj2[0]['y']),
            np.hypot(traj2[-1]['x'] - traj1[0]['x'], traj2[-1]['y'] - traj1[0]['y']),
            np.hypot(traj1[0]['x'] - traj2[0]['x'], traj1[0]['y'] - traj2[0]['y']),
            np.hypot(traj1[-1]['x'] - traj2[-1]['x'], traj1[-1]['y'] - traj2[-1]['y'])
        ]
        min_distance = min(distances)
        avg_speed = (meteor1['speed'] + meteor2['speed']) / 2
        max_distance = max(50, avg_speed * 10)

        if min_distance > max_distance:
            return False

        speed_ratio = min(meteor1['speed'], meteor2['speed']) / max(meteor1['speed'], meteor2['speed'], 0.01)
        if speed_ratio < 0.5:
            return False

        return True

    def merge_two_meteors(self, meteor1, meteor2):
        all_points = meteor1['trajectory'] + meteor2['trajectory']
        all_points.sort(key=lambda p: p['frame'])

        unique_points = []
        prev_frame = -1
        for point in all_points:
            if point['frame'] != prev_frame:
                unique_points.append(point)
                prev_frame = point['frame']
            elif point['brightness'] > unique_points[-1]['brightness']:
                unique_points[-1] = point

        xs = [p['x'] for p in unique_points]
        ys = [p['y'] for p in unique_points]

        merged = {
            'meteor_id': meteor1['meteor_id'],
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

        for spot in spots:
            cv2.circle(vis, (spot['x'], spot['y']), 5, (0, 255, 0), 2)

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
               f"Active: {len(self.active_tracks)} | Light Trail: {self.light_trail_mode}")
        cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)

        filename = self.debug_folder / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(filename), combined)

    # ==================== 主要處理流程（整合光鬼）====================
    def process_video(self):
        self.log("=" * 80)
        self.log("光鬼增強版流星偵測系統（亮背景友好參數整合）")
        self.log("=" * 80)
        self.log(f"影片: {self.video_path}")
        self.log(f"總幀數: {self.total_frames}, FPS: {self.fps:.2f}, proc_scale={self.proc_scale}")
        self.log(f"光鬼設定: length={self.light_trail.trail_length}, mode={self.light_trail_mode}")
        self.log(f"參數: movement_check={self.movement_check_frames}幀, min_movement={self.min_movement_pixels}px")
        if self.debug:
            self.log("調試模式: 開啟")

        cap = cv2.VideoCapture(str(self.video_path))
        ret, first_frame = cap.read()
        if not ret:
            self.log("❌ 無法讀取影片")
            return []

        prev_gray_full = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        prev_gray_full = cv2.GaussianBlur(prev_gray_full, (3, 3), 0)

        # 若使用縮放，先把縮小版加入 buffer
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
                delta_trail = cv2.absdiff(curr_gray, trail_gray)
                delta_trail = cv2.medianBlur(delta_trail, 3)
                delta = cv2.max(delta_standard, delta_trail)
            else:
                delta = delta_standard

            # 場景切換檢查
            scene_debug = {}
            is_scene_change = self.is_scene_change(delta_standard, debug_info=scene_debug)
            if is_scene_change:
                self.scene_change_frames.append(frame_idx)
                if self.debug:
                    self.log(f"  [場景切換] Frame {frame_idx} - change_ratio: {scene_debug.get('change_ratio', 0):.3f}")
                self.active_tracks.clear()
                self.light_trail.frame_buffer.clear()
                self.gray_buffer.clear()
                # 重新放入當前幀（縮放版）
                self.gray_buffer.append(curr_gray)
                prev_gray = curr_gray
                frame_idx += 1
                continue

            # 動態閾值（提高以減少亮背景噪音）
            dynamic_threshold = max(8, int(15 * np.sqrt(max(1, N))))
            spots = self.find_bright_spots(delta, threshold=dynamic_threshold, scale_up_factor=scale_up)

            # pending_spots 用於「穩定性檢查」
            # 存成原始座標 (x,y)
            self.pending_spots.append([(s['x'], s['y']) for s in spots])

            # 如果單幀 spots 過多，直接跳過該幀（保護）
            if len(spots) > self.spot_frame_cap:
                if self.debug:
                    self.log(f"[跳過] Frame {frame_idx} - spots {len(spots)} 太多，跳過該幀")
                frame_idx += 1
                continue

            if self.debug and (len(spots) > 0 or len(self.active_tracks) > 0):
                # 把 delta (縮放版) 與 curr_frame (原始) 傳入 debug
                self.save_debug_frame(curr_frame, cv2.resize(delta_standard, (self.width, self.height)) if self.proc_scale != 1.0 else delta_standard,
                                      cv2.resize(delta, (self.width, self.height)) if self.proc_scale != 1.0 else delta,
                                      spots, frame_idx)

            # 更新軌跡遺失計數
            for track in self.active_tracks:
                track['lost_frames'] += 1

            # 匹配點到軌跡（需檢查穩定性）
            matched_tracks = set()
            for spot in spots:
                # 只有穩定出現才能新建軌跡（避免一幀噪音）
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
                            f"幀 {meteor['frame_range'][0]}-{meteor['frame_range'][1]} "
                            f"(時間 {meteor['frame_range'][0]/self.fps:.1f}s-"
                            f"{meteor['frame_range'][1]/self.fps:.1f}s), "
                            f"長度 {meteor['length']:.1f}px, "
                            f"速度 {meteor['speed']:.1f}px/frame, "
                            f"點數 {meteor['duration_frames']}")

            frame_idx += 1
            if frame_idx % 100 == 0:
                self.log(f"處理進度: {frame_idx}/{self.total_frames}, Active: {len(self.active_tracks)}, Meteors: {len(self.meteors)}")

        # 處理剩餘軌跡
        for track in self.active_tracks:
            meteor = self.finalize_track(track)
            if meteor is not None:
                self.meteors.append(meteor)

        cap.release()

        # 合併重複偵測
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

    # ==================== 結果儲存 / 標記影片等 ====================
    def save_results(self):
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

                    label = f"{meteor['meteor_id']} | {meteor['speed']:.1f}px/f"
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
        description='光鬼增強版流星偵測系統 - Light Trail + Kalman',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python meteor_detector_light_trail.py --video me.mp4
  python meteor_detector_light_trail.py --video me.mp4 --trail-length 3 --trail-mode max --proc-scale 0.5 --debug
        """
    )
    parser.add_argument('--video', type=str, required=True, help='影片路徑')
    parser.add_argument('--trail-length', type=int, default=3,
                       help='光鬼累積幀數 (建議 3-8，預設 3)')
    parser.add_argument('--trail-mode', type=str, default='enhanced',
                       choices=['weighted', 'max', 'enhanced'],
                       help='光鬼模式 (預設 enhanced)')
    parser.add_argument('--proc-scale', type=float, default=1.0,
                       help='處理解析度縮放 (0.5 ~ 1.0)，0.5 可加速並降低噪音')
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
        light_trail_mode=args.trail_mode,
        proc_scale=args.proc_scale
    )
    tracker.process_video()


if __name__ == '__main__':
    main()
