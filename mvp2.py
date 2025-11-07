"""
流星偵測系統 - 增強版

新功能：
1. 靜止點過濾（10幀內不移動就剔除）
2. 直線運動模型（基於軌跡預測）
3. 快速/短流星偵測（多尺度檢測）

使用方法:
python meteor_detector_enhanced.py --video me.mp4 --debug
"""
import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict

class MeteorTracker:
    def __init__(self, video_path, output_folder, debug=False):
        self.video_path = Path(video_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        
        # 讀取影片資訊
        cap = cv2.VideoCapture(str(video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        self.meteors = []
        self.log_file = self.output_folder / "detection.log"
        self.active_tracks = []
        self.meteor_id_counter = 0
        self.scene_change_frames = []
        
        # === 新增參數 ===
        self.min_movement_pixels = 5  # 10幀內至少移動5像素
        self.movement_check_frames = 10  # 檢查移動的幀數窗口
        
        if debug:
            self.debug_folder = self.output_folder / "debug_frames"
            self.debug_folder.mkdir(exist_ok=True)
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def is_scene_change(self, delta, threshold_ratio=0.05):
        """偵測場景切換"""
        total_pixels = delta.shape[0] * delta.shape[1]
        changed_pixels = np.count_nonzero(delta > 20)
        change_ratio = changed_pixels / total_pixels
        return change_ratio > threshold_ratio
    
    def find_bright_spots(self, delta_frame, threshold=12):
        """
        在差分幀中找出明亮的移動點
        降低閾值以捕捉快速流星
        """
        # 二值化
        _, binary = cv2.threshold(delta_frame, threshold, 255, cv2.THRESH_BINARY)
        
        # 輕度降噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 找出連通區域
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # 降低面積限制以捕捉小而快的流星
            if 0.05 < area < 12000:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 計算亮度
                    mask = np.zeros_like(delta_frame)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    brightness = cv2.mean(delta_frame, mask=mask)[0]
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    spots.append({
                        'x': cx,
                        'y': cy,
                        'area': area,
                        'brightness': brightness,
                        'contour': contour,
                        'bbox': (x, y, w, h)
                    })
        
        return spots
    
    def predict_next_position(self, track):
        """
        基於直線運動模型預測下一個位置
        
        改進：使用最近3-5個點來計算平均速度
        """
        points = track['points']
        
        if len(points) < 2:
            # 沒有足夠點，返回最後位置
            return points[-1]['x'], points[-1]['y']
        
        # 使用最近的點計算速度（最多5個）
        recent_points = points[-min(5, len(points)):]
        
        # 計算平均速度
        vx_list = []
        vy_list = []
        
        for i in range(1, len(recent_points)):
            vx = recent_points[i]['x'] - recent_points[i-1]['x']
            vy = recent_points[i]['y'] - recent_points[i-1]['y']
            vx_list.append(vx)
            vy_list.append(vy)
        
        avg_vx = np.mean(vx_list)
        avg_vy = np.mean(vy_list)
        
        # 預測位置
        last_point = points[-1]
        predicted_x = last_point['x'] + avg_vx
        predicted_y = last_point['y'] + avg_vy
        
        # 儲存預測資訊
        track['predicted_velocity'] = (avg_vx, avg_vy)
        
        return predicted_x, predicted_y
    
    def check_trajectory_consistency(self, track, new_spot):
        """
        檢查新點是否符合軌跡的直線運動特性
        
        策略：
        1. 計算新點與預測位置的距離
        2. 檢查運動方向是否一致
        """
        if len(track['points']) < 2:
            return True  # 點數不夠，無法判斷
        
        # 預測位置
        pred_x, pred_y = self.predict_next_position(track)
        
        # 計算與預測位置的距離
        pred_distance = np.sqrt((new_spot['x'] - pred_x)**2 + 
                               (new_spot['y'] - pred_y)**2)
        
        # 如果有預測速度
        if 'predicted_velocity' in track:
            vx, vy = track['predicted_velocity']
            expected_speed = np.sqrt(vx**2 + vy**2)
            
            # 允許一定的偏差（30%）
            max_deviation = max(expected_speed * 1.5, 30)
            
            if pred_distance > max_deviation:
                if self.debug:
                    self.log(f"    [軌跡不連續] Track {track['track_id']}: "
                           f"預測偏差 {pred_distance:.1f} > {max_deviation:.1f}")
                return False
        
        # 檢查方向一致性（如果有足夠點）
        if len(track['points']) >= 3:
            # 計算過去的運動方向
            p1 = track['points'][-2]
            p2 = track['points'][-1]
            
            old_vx = p2['x'] - p1['x']
            old_vy = p2['y'] - p1['y']
            
            # 計算新的運動方向
            new_vx = new_spot['x'] - p2['x']
            new_vy = new_spot['y'] - p2['y']
            
            # 計算角度變化
            old_angle = np.arctan2(old_vy, old_vx)
            new_angle = np.arctan2(new_vy, new_vx)
            angle_diff = abs(old_angle - new_angle) * 180 / np.pi
            
            # 角度變化不應超過45度（流星是直線運動）
            if angle_diff > 45 and angle_diff < 315:
                if self.debug:
                    self.log(f"    [方向改變] Track {track['track_id']}: "
                           f"角度變化 {angle_diff:.1f}°")
                return False
        
        return True
    
    def match_spot_to_track(self, spot, max_distance=60):
        """
        將偵測到的點匹配到現有軌跡
        
        改進：優先考慮預測位置
        """
        best_track = None
        min_score = float('inf')
        
        for track in self.active_tracks:
            if len(track['points']) == 0:
                continue
            
            # 使用預測位置計算距離
            pred_x, pred_y = self.predict_next_position(track)
            
            pred_distance = np.sqrt((spot['x'] - pred_x)**2 + 
                                   (spot['y'] - pred_y)**2)
            
            # 計算匹配分數（距離越小越好）
            score = pred_distance
            
            # 加入時間懲罰（越久沒更新，分數越差）
            time_penalty = track['lost_frames'] * 5
            score += time_penalty
            
            if score < min_score and score < max_distance:
                # 檢查軌跡一致性
                if self.check_trajectory_consistency(track, spot):
                    min_score = score
                    best_track = track
        
        return best_track
    
    def check_movement(self, track):
        """
        檢查軌跡是否有足夠的移動
        
        新增：10幀內至少移動 min_movement_pixels 像素
        """
        points = track['points']
        
        if len(points) < 2:
            return True  # 點數太少，暫時保留
        
        # 取最近的 movement_check_frames 幀
        check_window = min(self.movement_check_frames, len(points))
        recent_points = points[-check_window:]
        
        # 計算這段時間內的總移動距離
        total_movement = 0
        for i in range(1, len(recent_points)):
            dx = recent_points[i]['x'] - recent_points[i-1]['x']
            dy = recent_points[i]['y'] - recent_points[i-1]['y']
            total_movement += np.sqrt(dx**2 + dy**2)
        
        # 計算起點到終點的直線距離
        start_point = recent_points[0]
        end_point = recent_points[-1]
        straight_distance = np.sqrt(
            (end_point['x'] - start_point['x'])**2 + 
            (end_point['y'] - start_point['y'])**2
        )
        
        # 檢查是否移動不足
        if straight_distance < self.min_movement_pixels:
            if self.debug:
                self.log(f"    [靜止點] Track {track['track_id']}: "
                       f"{check_window}幀內僅移動 {straight_distance:.1f}px")
            return False
        
        return True
    
    def create_new_track(self, spot, frame_idx):
        """建立新的軌跡"""
        track = {
            'track_id': self.meteor_id_counter,
            'points': [{'x': spot['x'], 'y': spot['y'], 
                       'frame': frame_idx, 'brightness': spot['brightness']}],
            'start_frame': frame_idx,
            'last_update': frame_idx,
            'lost_frames': 0,
            'predicted_velocity': (0, 0)
        }
        self.meteor_id_counter += 1
        return track
    
    def update_track(self, track, spot, frame_idx):
        """更新軌跡"""
        track['points'].append({
            'x': spot['x'], 
            'y': spot['y'], 
            'frame': frame_idx,
            'brightness': spot['brightness']
        })
        track['last_update'] = frame_idx
        track['lost_frames'] = 0
    #aaa
    def is_meteor_track(self, track):
        """
        驗證軌跡是否為流星
        
        改進：
        1. 降低最小點數要求（捕捉快速流星）
        2. 檢查移動
        3. 降低最小長度要求
        """
        points = track['points']
        
        # 降低最小點數要求（原本4，現在2）
        if len(points) < 4:
            return False
        
        # === 新增：檢查移動 ===
        if not self.check_movement(track):
            return False
        
        # 檢查是否在場景切換附近
        start_frame = track['start_frame']
        for scene_frame in self.scene_change_frames:
            if abs(start_frame - scene_frame) <= 3:
                return False
        
        # 計算軌跡長度
        total_length = 0
        for i in range(1, len(points)):
            dx = points[i]['x'] - points[i-1]['x']
            dy = points[i]['y'] - points[i-1]['y']
            total_length += np.sqrt(dx**2 + dy**2)
        
        # 降低最小長度要求（原本4，現在2）
        if total_length < 4:
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 軌跡太短 ({total_length:.1f}px)")
            return False
        
        # 計算速度
        duration = points[-1]['frame'] - points[0]['frame']
        if duration == 0:
            duration = 1
        speed = total_length / duration
        
        # 調整速度範圍（允許更快的流星）
        if speed < 1 or speed > 50000:  # 提高上限
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 速度超出範圍 ({speed:.1f}px/frame)")
            return False
        
        # 線性度檢查（只在有足夠點時）
        if len(points) >= 2:
            pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
            
            try:
                [vx, vy, cx, cy] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                
                line_point = np.array([cx[0], cy[0]])
                line_direction = np.array([vx[0], vy[0]])
                
                distances = []
                for pt in pts:
                    vec = pt - line_point
                    dist = np.abs(np.cross(line_direction, vec))
                    distances.append(dist)
                
                mean_deviation = np.mean(distances)
                
                # 放寬線性度要求
                if mean_deviation > 25.0:  # 原本15，現在25
                    if self.debug:
                        self.log(f"  [DEBUG] Track {track['track_id']}: 線性度不佳 ({mean_deviation:.1f})")
                    return False
            except:
                pass
        
        # 亮度檢查
        brightnesses = [p['brightness'] for p in points]
        max_brightness = max(brightnesses)
        
        if max_brightness < 0.5:  # 降低亮度要求
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 亮度太低 ({max_brightness:.1f})")
            return False
        
        return True
    #aaa
    def merge_overlapping_meteors(self):
        """
        合併重疊的流星軌跡
        
        改進：基於直線運動模型判斷是否為同一流星
        """
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
                
                # 檢查是否為同一流星
                if self.is_same_meteor(meteor1, meteor2):
                    group.append(meteor2)
                    group_indices.add(j)
            
            # 合併這組流星
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
        """
        判斷兩個軌跡是否為同一顆流星
        
        改進：基於軌跡的延伸判斷
        """
        # 時間重疊或接近
        r1 = meteor1['frame_range']
        r2 = meteor2['frame_range']
        
        # 允許最多3幀的間隔
        time_gap = max(0, r2[0] - r1[1], r1[0] - r2[1])
        if time_gap > 3:
            return False
        
        # 計算軌跡的延伸線
        traj1 = meteor1['trajectory']
        traj2 = meteor2['trajectory']
        
        # 使用線性擬合
        if len(traj1) >= 2 and len(traj2) >= 2:
            # 擬合直線
            pts1 = np.array([[p['x'], p['y']] for p in traj1], dtype=np.float32)
            pts2 = np.array([[p['x'], p['y']] for p in traj2], dtype=np.float32)
            
            try:
                [vx1, vy1, cx1, cy1] = cv2.fitLine(pts1, cv2.DIST_L2, 0, 0.01, 0.01)
                [vx2, vy2, cx2, cy2] = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # 計算方向相似度
                dot_product = vx1*vx2 + vy1*vy2
                angle_similarity = abs(dot_product[0])  # cos(角度差)
                
                # 方向應該相似（cos > 0.9 即角度差 < 25度）
                if angle_similarity < 0.9:
                    return False
                
                # 檢查軌跡是否在同一條直線上
                # 計算軌跡2的點到軌跡1直線的距離
                line_point = np.array([cx1[0], cy1[0]])
                line_direction = np.array([vx1[0], vy1[0]])
                
                distances = []
                for pt in pts2:
                    vec = pt - line_point
                    dist = np.abs(np.cross(line_direction, vec))
                    distances.append(dist)
                
                mean_dist = np.mean(distances)
                
                # 平均距離應該很小（在同一條線上）
                if mean_dist > 20:
                    return False
                
                return True
                
            except:
                pass
        
        # 如果無法擬合直線，用簡單的距離判斷
        min_dist = float('inf')
        for p1 in traj1:
            for p2 in traj2:
                dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                min_dist = min(min_dist, dist)
        
        return min_dist < 15
    
    def merge_meteor_group(self, group):
        """合併一組流星軌跡"""
        all_points = []
        for meteor in group:
            all_points.extend(meteor['trajectory'])
        
        all_points.sort(key=lambda p: p['frame'])
        
        # 去除重複幀
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
            'bbox': [max(0, min(xs)-10), max(0, min(ys)-10), 
                    max(xs)-min(xs)+20, max(ys)-min(ys)+20],
            'length': self.calculate_trajectory_length(unique_points),
            'angle': self.calculate_trajectory_angle(unique_points),
            'speed': self.calculate_average_speed(unique_points),
            'max_brightness': max([p['brightness'] for p in unique_points])
        }
        
        return merged
    
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
            'bbox': [max(0, min(xs)-10), max(0, min(ys)-10), 
                    max(xs)-min(xs)+20, max(ys)-min(ys)+20],
            'length': self.calculate_trajectory_length(points),
            'angle': self.calculate_trajectory_angle(points),
            'speed': self.calculate_average_speed(points),
            'max_brightness': max([p['brightness'] for p in points])
        }
        
        return meteor
    
    def calculate_trajectory_length(self, points):
        """計算軌跡總長度"""
        length = 0
        for i in range(1, len(points)):
            dx = points[i]['x'] - points[i-1]['x']
            dy = points[i]['y'] - points[i-1]['y']
            length += np.sqrt(dx**2 + dy**2)
        return length
    
    def calculate_trajectory_angle(self, points):
        """計算軌跡角度"""
        if len(points) < 2:
            return 0
        dx = points[-1]['x'] - points[0]['x']
        dy = points[-1]['y'] - points[0]['y']
        return np.arctan2(dy, dx) * 180 / np.pi
    
    def calculate_average_speed(self, points):
        """計算平均速度"""
        length = self.calculate_trajectory_length(points)
        duration = points[-1]['frame'] - points[0]['frame']
        if duration == 0:
            duration = 1
        return length / duration
    
    def save_debug_frame(self, frame, delta, spots, frame_idx, is_scene_change=False):
        """儲存調試幀"""
        if not self.debug or frame_idx % 25 != 0:
            return
        
        vis = frame.copy()
        delta_color = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
        
        # 標記所有偵測到的點
        for spot in spots:
            cv2.circle(vis, (spot['x'], spot['y']), 5, (0, 255, 0), 2)
        
        # 標記所有活躍軌跡（用不同顏色）
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255), (128, 255, 0)]
        for idx, track in enumerate(self.active_tracks):
            color = colors[idx % len(colors)]
            points = track['points']
            
            # 畫軌跡
            for i in range(1, len(points)):
                p1 = (points[i-1]['x'], points[i-1]['y'])
                p2 = (points[i]['x'], points[i]['y'])
                cv2.line(vis, p1, p2, color, 2)
            
            # 標記預測位置
            if len(points) >= 2:
                pred_x, pred_y = self.predict_next_position(track)
                cv2.circle(vis, (int(pred_x), int(pred_y)), 3, color, -1)
                cv2.circle(vis, (int(pred_x), int(pred_y)), 8, color, 1)
        
        combined = np.hstack([vis, delta_color])
        
        scene_text = " [場景切換]" if is_scene_change else ""
        info = f"Frame {frame_idx} | Spots: {len(spots)} | Active: {len(self.active_tracks)}{scene_text}"
        cv2.putText(combined, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255) if not is_scene_change else (0, 0, 255), 2)
        
        filename = self.debug_folder / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(filename), combined)
    
    def process_video(self):
        """處理整個影片"""
        self.log("="*80)
        self.log("流星偵測系統 - 增強版")
        self.log("="*80)
        self.log(f"影片: {self.video_path}")
        self.log(f"總幀數: {self.total_frames}, FPS: {self.fps:.2f}")
        self.log(f"參數: 移動檢查={self.movement_check_frames}幀, 最小移動={self.min_movement_pixels}px")
        if self.debug:
            self.log("調試模式: 開啟")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        ret, prev_frame = cap.read()
        if not ret:
            self.log("❌ 無法讀取影片")
            return
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (3, 3), 0)
        
        frame_idx = 1
        lost_track_threshold = 4  # 降低閾值以捕捉快速流星
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0)
            
            delta = cv2.absdiff(curr_gray, prev_gray)
            
            # 檢查場景切換
            is_scene_change = self.is_scene_change(delta)
            if is_scene_change:
                self.scene_change_frames.append(frame_idx)
                if self.debug:
                    self.log(f"  [場景切換] Frame {frame_idx}")
                self.active_tracks.clear()
                prev_gray = curr_gray
                frame_idx += 1
                continue
            
            # 找出明亮的移動點
            spots = self.find_bright_spots(delta, threshold=12)
            
            # 調試輸出
            if self.debug and (len(spots) > 0 or len(self.active_tracks) > 0):
                self.save_debug_frame(curr_frame, delta, spots, frame_idx, is_scene_change)
            
            # 更新軌跡
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
                if track['lost_frames'] > lost_track_threshold:
                    self.active_tracks.remove(track)
                    completed_tracks.append(track)
            
            # 驗證並儲存流星
            for track in completed_tracks:
                meteor = self.finalize_track(track)
                if meteor is not None:
                    self.meteors.append(meteor)
                    self.log(f"✓ 流星 #{len(self.meteors)}: "
                           f"幀 {meteor['frame_range'][0]}-{meteor['frame_range'][1]} "
                           f"(時間 {meteor['frame_range'][0]/self.fps:.1f}s-{meteor['frame_range'][1]/self.fps:.1f}s), "
                           f"長度 {meteor['length']:.1f}px, "
                           f"速度 {meteor['speed']:.1f}px/frame, "
                           f"點數 {meteor['duration_frames']}")
            
            prev_gray = curr_gray
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
        
        # 合併重疊的流星
        original_count = len(self.meteors)
        self.merge_overlapping_meteors()
        
        if original_count != len(self.meteors):
            self.log(f"\n軌跡合併: {original_count} → {len(self.meteors)} 顆流星")
        
        self.log(f"\n總共偵測到 {len(self.meteors)} 顆流星")
        self.log(f"場景切換次數: {len(self.scene_change_frames)}")
        
        # 儲存結果
        self.save_results()
        
        # 生成標記影片
        if len(self.meteors) > 0:
            self.create_annotated_video()
        else:
            self.log("\n⚠️ 未偵測到流星")
            if not self.debug:
                self.log("建議使用 --debug 參數重新執行以查看偵測細節")
        
        return self.meteors
    
    def save_results(self):
        """儲存偵測結果"""
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

def main():
    parser = argparse.ArgumentParser(description='流星偵測 - 增強版')
    parser.add_argument('--video', type=str, required=True, help='影片路徑')
    parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ 找不到影片: {args.video}")
        return
    
    output_folder = Path("output_enhanced") / Path(args.video).stem
    tracker = MeteorTracker(args.video, output_folder, debug=args.debug)
    tracker.process_video()

if __name__ == "__main__":
    main()