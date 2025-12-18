#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合版：MeteorTracker（含 Kalman、multi-frame 差分、RANSAC 短軌跡驗證、即時合併機制）
使用方式:
    python meteor_tracker_enhanced_kalman.py --video me.mp4 --debug

說明：本檔案為單一可執行檔，會讀取影片並輸出偵測結果與標註影片。
設計重點：
 - 時間感知（使用 dt = seconds）來處理低 FPS / frame-skip
 - SimpleKalman：用於平滑位置與預測
 - multi-frame 差分（依時間窗口選擇 N 幀差分）
 - 動態閾值、RANSAC 共線檢查、軌跡合併（含即時合併避免重複偵測）

 是否可以將這兩種版本的程式結合，並達到更棒更好的偵測效果
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
import random


# ------------------ Simple Kalman ------------------
class SimpleKalman:
    """
    極簡 Kalman filter for constant-velocity model
    state: [x, y, vx, vy]
    """
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
        # deep copy kalman state (安全的複製)
        new = SimpleKalman(0, 0)
        new.x = self.x.copy()
        new.P = self.P.copy()
        new.process_var = float(self.process_var)
        new.R = self.R.copy()
        return new


# ------------------ RANSAC 共線檢查 ------------------
def is_collinear_ransac(points, threshold=3.0, min_inliers_ratio=0.6, max_iterations=120):
    """
    對少數點做共線性檢驗（用於短軌跡）
    points: list of {'x':..,'y':..}
    """
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
        # 距離計算（點到線的垂直距離）
        distances = np.abs(np.cross(v, pts - p1)) / norm_v
        inliers = np.count_nonzero(distances < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            # 早停
            if best_inliers / n >= min_inliers_ratio:
                return True
    return (best_inliers / n) >= min_inliers_ratio


# ------------------ MeteorTracker 類別 ------------------
class MeteorTracker:
    def __init__(self, video_path, output_folder, debug=False):
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

        # 偵測結果
        self.meteors = []
        self.log_file = self.output_folder / "detection.log"
        self.active_tracks = []
        self.meteor_id_counter = 0
        self.scene_change_frames = []

        # 參數（可調整）
        self.min_movement_pixels = 5
        self.movement_check_frames = 10
        self.lost_track_threshold = 4

        # 多幀差分設定
        self.target_window_seconds = 0.4  # 可參考調整（低 FPS 增大）
        self.gray_buffer = deque(maxlen=max(30, int(self.fps * 2)))

        # 最近完成的軌跡緩衝（避免場景切換後重複偵測）
        self.recent_finalized = deque(maxlen=120)

        # debug 輸出
        if debug:
            self.debug_folder = self.output_folder / "debug_frames"
            self.debug_folder.mkdir(exist_ok=True)

        # log 檔案初始化
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Log start: {datetime.now()}")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "")

    def is_scene_change(self, delta, threshold_ratio=0.05):
        total_pixels = delta.shape[0] * delta.shape[1]
        changed_pixels = np.count_nonzero(delta > 20)
        change_ratio = changed_pixels / float(total_pixels)
        return change_ratio > threshold_ratio

    def find_bright_spots(self, delta_frame, threshold=12):
        # 二值化
        _, binary = cv2.threshold(delta_frame, int(threshold), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 0.05 < area < 12000:
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

    # ------------------ 新增：軌跡相似度工具 ------------------
    def trajectory_overlap_fraction(self, traj1, traj2, spatial_tol=12):
        """
        計算兩條軌跡在時間與空間上的重疊比例（以較短軌跡為基準）
        """
        map1 = {p['frame']: (p['x'], p['y']) for p in traj1}
        map2 = {p['frame']: (p['x'], p['y']) for p in traj2}
        common_frames = set(map1.keys()) & set(map2.keys())
        if not common_frames:
            matched = 0
            total = min(len(traj1), len(traj2))
            if total == 0:
                return 0.0
            for p in traj1:
                f = p['frame']
                for df in (-1, 0, 1):
                    ff = f + df
                    if ff in map2:
                        x1, y1 = p['x'], p['y']
                        x2, y2 = map2[ff]
                        if np.hypot(x1-x2, y1-y2) <= spatial_tol:
                            matched += 1
                            break
            return matched / float(total)
        else:
            matched = 0
            for f in common_frames:
                x1, y1 = map1[f]
                x2, y2 = map2[f]
                if np.hypot(x1-x2, y1-y2) <= spatial_tol:
                    matched += 1
            total = min(len(traj1), len(traj2))
            return matched / float(total) if total > 0 else 0.0

    def brightness_profile_correlation(self, traj1, traj2):
        map1 = {p['frame']: p.get('brightness', 0.0) for p in traj1}
        map2 = {p['frame']: p.get('brightness', 0.0) for p in traj2}
        common_frames = sorted(set(map1.keys()) & set(map2.keys()))
        if len(common_frames) < 3:
            return 0.0
        arr1 = np.array([map1[f] for f in common_frames], dtype=float)
        arr2 = np.array([map2[f] for f in common_frames], dtype=float)
        if np.std(arr1) < 1e-6 or np.std(arr2) < 1e-6:
            return 0.0
        corr = np.corrcoef((arr1 - arr1.mean()) / arr1.std(), (arr2 - arr2.mean()) / arr2.std())[0,1]
        return float(corr)

    def add_or_merge_meteor(self, new_meteor, time_gap_allow=4, angle_cos_threshold=0.92, overlap_threshold=0.35, brightness_corr_threshold=0.4):
        """
        嘗試把 new_meteor 與已存在的 meteors 合併（即時合併，避免重複偵測）
        回傳 True 表示已合併，False 表示加入新的
        """
        best_idx = None
        best_score = -1.0
        for i, existing in enumerate(self.meteors):
            r1 = existing['frame_range']
            r2 = new_meteor['frame_range']
            time_gap = max(0, r2[0] - r1[1], r1[0] - r2[1])
            if time_gap > time_gap_allow:
                continue
            ang1 = existing.get('angle', 0.0)
            ang2 = new_meteor.get('angle', 0.0)
            v1 = np.array([np.cos(np.deg2rad(ang1)), np.sin(np.deg2rad(ang1))])
            v2 = np.array([np.cos(np.deg2rad(ang2)), np.sin(np.deg2rad(ang2))])
            cos_sim = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
            if cos_sim < angle_cos_threshold:
                continue
            overlap = self.trajectory_overlap_fraction(existing['trajectory'], new_meteor['trajectory'], spatial_tol=12)
            if overlap < overlap_threshold:
                bright_corr = self.brightness_profile_correlation(existing['trajectory'], new_meteor['trajectory'])
                if bright_corr < brightness_corr_threshold:
                    continue
                score = 0.5 * bright_corr + 0.5 * overlap
            else:
                score = 0.8 * overlap + 0.2 * cos_sim
            if score > best_score:
                best_score = score
                best_idx = i
        # 也比對 recent_finalized（剛完成的短期軌跡）
        for j, existing in enumerate(list(self.recent_finalized)):
            r1 = existing['frame_range']
            r2 = new_meteor['frame_range']
            time_gap = max(0, r2[0] - r1[1], r1[0] - r2[1])
            if time_gap > time_gap_allow:
                continue
            ang1 = existing.get('angle', 0.0)
            ang2 = new_meteor.get('angle', 0.0)
            v1 = np.array([np.cos(np.deg2rad(ang1)), np.sin(np.deg2rad(ang1))])
            v2 = np.array([np.cos(np.deg2rad(ang2)), np.sin(np.deg2rad(ang2))])
            cos_sim = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
            if cos_sim < angle_cos_threshold:
                continue
            overlap = self.trajectory_overlap_fraction(existing['trajectory'], new_meteor['trajectory'], spatial_tol=12)
            if overlap < overlap_threshold:
                bright_corr = self.brightness_profile_correlation(existing['trajectory'], new_meteor['trajectory'])
                if bright_corr < brightness_corr_threshold:
                    continue
                score = 0.5 * bright_corr + 0.5 * overlap
            else:
                score = 0.8 * overlap + 0.2 * cos_sim
            if score > best_score:
                best_score = score
                best_idx = None  # indicate merge with recent_finalized
                best_recent = j
                best_existing = existing
        if best_idx is not None:
            merged = self.merge_meteor_group([self.meteors[best_idx], new_meteor])
            self.meteors[best_idx] = merged
            if self.debug:
                self.log(f"[即時合併] 新流星與現有 meteor_{best_idx} 合併 (score={best_score:.2f})")
            # push merged to recent_finalized
            self.recent_finalized.append(self.meteors[best_idx])
            return True
        # if merged with recent_finalized
        if 'best_existing' in locals() and best_existing is not None:
            merged = self.merge_meteor_group([best_existing, new_meteor])
            # replace in recent_finalized the merged one
            # find and replace
            for idx, item in enumerate(self.recent_finalized):
                if item['frame_range'] == best_existing['frame_range'] and item['max_brightness'] == best_existing.get('max_brightness'):
                    self.recent_finalized[idx] = merged
                    break
            # also push merged to main meteors list
            self.meteors.append(merged)
            if self.debug:
                self.log(f"[即時合併] 新流星與 recent_finalized 合併")
            return True
        # 若沒合併，直接加入
        self.meteors.append(new_meteor)
        self.recent_finalized.append(new_meteor)
        return False

    def create_new_track(self, spot, frame_idx):
        curr_time = frame_idx / max(1.0, self.fps)
        kalman = SimpleKalman(spot['x'], spot['y'], init_vx=0.0, init_vy=0.0, process_var=2.0, meas_var=6.0)
        track = {
            'track_id': self.meteor_id_counter,
            'points': [{'x': spot['x'], 'y': spot['y'], 'frame': frame_idx, 'brightness': spot['brightness']}],
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

        # Kalman predict + update
        track['kalman'].predict(dt)
        kstate = track['kalman'].update(spot['x'], spot['y'])
        track['predicted_velocity'] = (float(kstate[2]), float(kstate[3]))

        track['points'].append({
            'x': int(kstate[0]), 'y': int(kstate[1]), 'frame': frame_idx, 'brightness': spot['brightness']
        })
        track['last_update'] = frame_idx
        track['lost_frames'] = 0
        track['last_time'] = curr_time

    def predict_next_position(self, track, predict_ahead_seconds=None):
        # 若未指定預測時間，預測下一幀
        if predict_ahead_seconds is None:
            predict_ahead_seconds = 1.0 / max(1.0, self.fps)
        try:
            # 複製 kalman 以避免更動原本狀態
            kclone = track['kalman'].clone()
            kclone.predict(predict_ahead_seconds)
            pred = kclone.x
            return float(pred[0]), float(pred[1])
        except Exception:
            last = track['points'][-1]
            return last['x'], last['y']

    def check_trajectory_consistency(self, track, new_spot):
        if len(track['points']) < 2:
            return True
        pred_x, pred_y = self.predict_next_position(track)
        pred_distance = np.hypot(new_spot['x'] - pred_x, new_spot['y'] - pred_y)
        if 'predicted_velocity' in track:
            vx, vy = track['predicted_velocity']
            expected_speed = np.hypot(vx, vy)
            max_deviation = max(expected_speed * 1.5, 30)
            if pred_distance > max_deviation:
                if self.debug:
                    self.log(f"    [軌跡不連續] Track {track['track_id']}: 預測偏差 {pred_distance:.1f} > {max_deviation:.1f}")
                return False

        if len(track['points']) >= 3:
            p1 = track['points'][-2]
            p2 = track['points'][-1]
            old_vx = p2['x'] - p1['x']
            old_vy = p2['y'] - p1['y']
            new_vx = new_spot['x'] - p2['x']
            new_vy = new_spot['y'] - p2['y']
            old_angle = np.arctan2(old_vy, old_vx)
            new_angle = np.arctan2(new_vy, new_vx)
            angle_diff = abs(old_angle - new_angle) * 180 / np.pi
            if angle_diff > 45 and angle_diff < 315:
                if self.debug:
                    self.log(f"    [方向改變] Track {track['track_id']}: 角度變化 {angle_diff:.1f}°")
                return False
        return True

    def match_spot_to_track(self, spot, max_distance=60):
        best_track = None
        min_score = float('inf')
        for track in self.active_tracks:
            if len(track['points']) == 0:
                continue
            pred_x, pred_y = self.predict_next_position(track)
            pred_distance = np.hypot(spot['x'] - pred_x, spot['y'] - pred_y)
            time_penalty = track['lost_frames'] * 5
            score = pred_distance + time_penalty
            # 動態 max_distance：依照失落時間、解析度放寬
            dynamic_max = max_distance + int(track['lost_frames'] * 8)
            if score < min_score and score < dynamic_max:
                if self.check_trajectory_consistency(track, spot):
                    min_score = score
                    best_track = track
        return best_track

    def check_movement(self, track):
        points = track['points']
        if len(points) < 2:
            return True
        check_window = min(self.movement_check_frames, len(points))
        recent_points = points[-check_window:]
        start_point = recent_points[0]
        end_point = recent_points[-1]
        straight_distance = np.hypot(end_point['x'] - start_point['x'], end_point['y'] - start_point['y'])
        if straight_distance < self.min_movement_pixels:
            if self.debug:
                self.log(f"    [靜止點] Track {track['track_id']}: {check_window}幀內僅移動 {straight_distance:.1f}px")
            return False
        return True

    def is_meteor_track(self, track):
        points = track['points']
        # 降低最小點數門檻，但短軌跡需通過共線或亮度檢查
        if len(points) < 4:
            return False
        if len(points) < 4:
            # 用 RANSAC 共線檢查
            if not is_collinear_ransac(points, threshold=4.0, min_inliers_ratio=0.6):
                if self.debug:
                    self.log(f"  [DEBUG] Track {track['track_id']}: 少點且非共線")
                return False
        # 檢查移動
        if not self.check_movement(track):
            return False
        # 場景切換檢查
        start_frame = track['start_frame']
        for scene_frame in self.scene_change_frames:
            if abs(start_frame - scene_frame) <= 3:
                return False
        # 計算軌跡長度
        total_length = self.calculate_trajectory_length(points)
        if total_length < 4:
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 軌跡太短 ({total_length:.1f}px)")
            return False
        # 速度檢查
        duration = points[-1]['frame'] - points[0]['frame']
        if duration == 0:
            duration = 1
        speed = total_length / duration
        if speed < 0.5 or speed > 50000:
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 速度超出範圍 ({speed:.1f}px/frame)")
            return False
        # 線性度檢查（放寬）
        if len(points) >= 2:
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
                if mean_deviation > 25.0:
                    if self.debug:
                        self.log(f"  [DEBUG] Track {track['track_id']}: 線性度不佳 ({mean_deviation:.1f})")
                    return False
            except Exception:
                pass
        # 亮度檢查
        brightnesses = [p['brightness'] for p in points]
        max_brightness = max(brightnesses)
        if max_brightness < 0.5:
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 亮度太低 ({max_brightness:.1f})")
            return False
        return True

    def merge_overlapping_meteors(self):
        if len(self.meteors) <= 1:
            return
        merged = []
        used = set()
        for i, meteor1 in enumerate(self.meteors):
            if i in used:
                continue
            group = [meteor1]
            group_indices = {i}
            for j, meteor2 in enumerate(self.meteors):
                if j <= i or j in used:
                    continue
                if self.is_same_meteor(meteor1, meteor2):
                    group.append(meteor2)
                    group_indices.add(j)
            if len(group) > 1:
                merged_meteor = self.merge_meteor_group(group)
                merged.append(merged_meteor)
                used.update(group_indices)
                if self.debug:
                    self.log(f"  [合併] {len(group)} 個軌跡合併為 {merged_meteor['meteor_id']}")
            else:
                merged.append(meteor1)
                used.add(i)
        self.meteors = merged

    def is_same_meteor(self, meteor1, meteor2):
        r1 = meteor1['frame_range']
        r2 = meteor2['frame_range']
        time_gap = max(0, r2[0] - r1[1], r1[0] - r2[1])
        if time_gap > 3:
            return False
        traj1 = meteor1['trajectory']
        traj2 = meteor2['trajectory']
        if len(traj1) >= 2 and len(traj2) >= 2:
            pts1 = np.array([[p['x'], p['y']] for p in traj1], dtype=np.float32)
            pts2 = np.array([[p['x'], p['y']] for p in traj2], dtype=np.float32)
            try:
                [vx1, vy1, cx1, cy1] = cv2.fitLine(pts1, cv2.DIST_L2, 0, 0.01, 0.01)
                [vx2, vy2, cx2, cy2] = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01)
                dot_product = vx1 * vx2 + vy1 * vy2
                angle_similarity = abs(dot_product[0])
                if angle_similarity < 0.9:
                    return False
                line_point = np.array([cx1[0], cy1[0]])
                line_direction = np.array([vx1[0], vy1[0]])
                distances = []
                for pt in pts2:
                    vec = pt - line_point
                    dist = abs(np.cross(line_direction, vec))
                    distances.append(dist)
                mean_dist = np.mean(distances)
                if mean_dist > 20:
                    return False
                return True
            except Exception:
                pass
        min_dist = float('inf')
        for p1 in traj1:
            for p2 in traj2:
                dist = np.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
                min_dist = min(min_dist, dist)
        return min_dist < 15

    def merge_meteor_group(self, group):
        all_points = []
        for meteor in group:
            all_points.extend(meteor['trajectory'])
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
            'meteor_id': group[0]['meteor_id'],
            'frame_range': [unique_points[0]['frame'], unique_points[-1]['frame']],
            'duration_frames': len(unique_points),
            'trajectory': unique_points,
            'bbox': [max(0, min(xs) - 10), max(0, min(ys) - 10), max(xs) - min(xs) + 20, max(ys) - min(ys) + 20],
            'length': self.calculate_trajectory_length(unique_points),
            'angle': self.calculate_trajectory_angle(unique_points),
            'speed': self.calculate_average_speed(unique_points),
            'max_brightness': max([p['brightness'] for p in unique_points])
        }
        return merged

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
            'bbox': [max(0, min(xs) - 10), max(0, min(ys) - 10), max(xs) - min(xs) + 20, max(ys) - min(ys) + 20],
            'length': self.calculate_trajectory_length(points),
            'angle': self.calculate_trajectory_angle(points),
            'speed': self.calculate_average_speed(points),
            'max_brightness': max([p['brightness'] for p in points])
        }
        return meteor

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

    def save_debug_frame(self, frame, delta, spots, frame_idx, is_scene_change=False):
        if not self.debug or frame_idx % 25 != 0:
            return
        vis = frame.copy()
        delta_color = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
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
                cv2.circle(vis, (int(pred_x), int(pred_y)), 8, color, 1)
        combined = np.hstack([vis, delta_color])
        scene_text = " [場景切換]" if is_scene_change else ""
        info = f"Frame {frame_idx} | Spots: {len(spots)} | Active: {len(self.active_tracks)}{scene_text}"
        cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        filename = self.debug_folder / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(filename), combined)

    def process_video(self):
        self.log("=" * 80)
        self.log("流星偵測系統 - Kalman + Multi-frame 差分")
        self.log("=" * 80)
        self.log(f"影片: {self.video_path}")
        self.log(f"總幀數: {self.total_frames}, FPS: {self.fps:.2f}")
        self.log(f"參數: movement_check={self.movement_check_frames} 幀, min_movement={self.min_movement_pixels}px")
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

        frame_idx = 1
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0)

            # push 到 buffer
            self.gray_buffer.append(curr_gray)

            N = max(1, int(round(self.target_window_seconds * self.fps)))
            if len(self.gray_buffer) >= N:
                prev_gray_k = self.gray_buffer[-N]
            else:
                prev_gray_k = self.gray_buffer[0]

            delta = cv2.absdiff(curr_gray, prev_gray_k)

            # 場景切換檢查
            is_scene_change = self.is_scene_change(delta)
            if is_scene_change:
                self.scene_change_frames.append(frame_idx)
                if self.debug:
                    self.log(f"  [場景切換] Frame {frame_idx}")
                self.active_tracks.clear()
                prev_gray = curr_gray
                frame_idx += 1
                continue

            # 動態閾值
            dynamic_threshold = max(8, int(12 * np.sqrt(max(1, N))))
            spots = self.find_bright_spots(delta, threshold=dynamic_threshold)

            if self.debug and (len(spots) > 0 or len(self.active_tracks) > 0):
                self.save_debug_frame(curr_frame, delta, spots, frame_idx, is_scene_change)

            for track in self.active_tracks:
                track['lost_frames'] += 1

            matched_tracks = set()
            for spot in spots:
                track = self.match_spot_to_track(spot)
                if track is not None and track['track_id'] not in matched_tracks:
                    self.update_track(track, spot, frame_idx)
                    matched_tracks.add(track['track_id'])
                else:
                    new_track = self.create_new_track(spot, frame_idx)
                    self.active_tracks.append(new_track)

            completed_tracks = []
            for track in self.active_tracks[:]:
                if track['lost_frames'] > self.lost_track_threshold:
                    self.active_tracks.remove(track)
                    completed_tracks.append(track)

            for track in completed_tracks:
                meteor = self.finalize_track(track)
                if meteor is not None:
                    # 立即嘗試合併或加入，避免重複偵測
                    self.add_or_merge_meteor(meteor)
                    self.log(f"✓ 流星 #{len(self.meteors)}: 幀 {meteor['frame_range'][0]}-{meteor['frame_range'][1]} (時間 {meteor['frame_range'][0]/self.fps:.1f}s-{meteor['frame_range'][1]/self.fps:.1f}s), 長度 {meteor['length']:.1f}px, 速度 {meteor['speed']:.1f}px/frame, 點數 {meteor['duration_frames']}")

            frame_idx += 1
            if frame_idx % 100 == 0:
                self.log(f"處理進度: {frame_idx}/{self.total_frames}, 活躍軌跡: {len(self.active_tracks)}, 已偵測流星: {len(self.meteors)}")

        # 處理剩餘軌跡
        for track in self.active_tracks:
            meteor = self.finalize_track(track)
            if meteor is not None:
                self.add_or_merge_meteor(meteor)

        cap.release()

        original_count = len(self.meteors)
        self.merge_overlapping_meteors()
        if original_count != len(self.meteors):
            self.log(f"軌跡合併: {original_count} → {len(self.meteors)} 顆流星")

        self.log(f"總共偵測到 {len(self.meteors)} 顆流星")
        self.log(f"場景切換次數: {len(self.scene_change_frames)}")

        self.save_results()
        if len(self.meteors) > 0:
            self.create_annotated_video()
        else:
            self.log("⚠️ 未偵測到流星")
            if not self.debug:
                self.log("建議使用 --debug 參數重新執行以查看偵測細節")
        return self.meteors

    def save_results(self):
        result = {
            'video': str(self.video_path),
            'total_meteors': len(self.meteors),
            'fps': self.fps,
            'scene_changes': self.scene_change_frames,
            'meteors': self.meteors
        }
        with open(self.output_folder / "results.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        self.log(f"✓ 結果已儲存: {self.output_folder / 'results.json'}")

    def create_annotated_video(self):
        self.log("開始生成標記影片...")
        output_video = self.output_folder / f"{self.video_path.stem}_annotated.mp4"
        cap = cv2.VideoCapture(str(self.video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, self.fps, (self.width, self.height))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in self.scene_change_frames:
                cv2.putText(frame, "SCENE CHANGE", (self.width//2-100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
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
                            cv2.circle(frame, (point['x'], point['y']), 8, (0, 255, 255), 2)
                            cv2.circle(frame, (point['x'], point['y']), 3, (0, 0, 255), -1)
                    bbox = meteor['bbox']
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 1)
                    label = f"{meteor['meteor_id']} | {meteor['speed']:.1f}px/f"
                    cv2.putText(frame, label, (bbox[0], max(15, bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            time_sec = frame_idx / self.fps
            info = f"Frame: {frame_idx} | Time: {time_sec:.1f}s | Meteors: {len(self.meteors)}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                self.log(f"  標記進度: {frame_idx}/{self.total_frames}")
        cap.release()
        out.release()
        self.log(f"✓ 標記影片已儲存: {output_video}")
        return output_video


# ------------------ CLI ------------------
def main():
    parser = argparse.ArgumentParser(description='流星偵測 - Kalman + Multi-frame')
    parser.add_argument('--video', type=str, required=True, help='影片路徑')
    parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    args = parser.parse_args()
    if not os.path.exists(args.video):
        print(f"❌ 找不到影片: {args.video}")
        return
    output_folder = Path("output_enhanced_kalman") / Path(args.video).stem
    tracker = MeteorTracker(args.video, output_folder, debug=args.debug)
    tracker.process_video()


if __name__ == '__main__':
    main()
