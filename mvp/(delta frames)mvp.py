"""
流星偵測系統 - Delta Frame 優化版

改進:
1. 增加場景切換偵測，避免誤判
2. 改進軌跡合併，避免同一流星被標記多次
3. 更精確的流星特徵驗證

使用方法:
python meteor_detector_delta.py --video me.mp4 --debug
"""
import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict, deque

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
        self.scene_change_frames = []  # 記錄場景切換幀
        
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
        """
        偵測是否為場景切換
        場景切換時會有大量像素變化
        """
        # 計算變化的像素比例
        total_pixels = delta.shape[0] * delta.shape[1]
        changed_pixels = np.count_nonzero(delta > 20)
        change_ratio = changed_pixels / total_pixels
        
        # 如果超過30%的像素有變化，可能是場景切換
        return change_ratio > threshold_ratio
    
    def find_bright_spots(self, delta_frame, threshold=15):
        """在差分幀中找出明亮的移動點"""
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
            # 流星通常是小而亮的點
            if 0.1 < area < 10000:  # 降低上限避免大範圍變化
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
    
    def match_spot_to_track(self, spot, max_distance=80):
        """將偵測到的點匹配到現有軌跡"""
        best_track = None
        min_distance = max_distance
        
        for track in self.active_tracks:
            if len(track['points']) > 0:
                last_point = track['points'][-1]
                distance = np.sqrt((spot['x'] - last_point['x'])**2 + 
                                 (spot['y'] - last_point['y'])**2)
                
                # 預測下一個位置
                if len(track['points']) >= 2:
                    vx = track['points'][-1]['x'] - track['points'][-2]['x']
                    vy = track['points'][-1]['y'] - track['points'][-2]['y']
                    predicted_x = last_point['x'] + vx
                    predicted_y = last_point['y'] + vy
                    predicted_distance = np.sqrt((spot['x'] - predicted_x)**2 + 
                                                (spot['y'] - predicted_y)**2)
                    distance = min(distance, predicted_distance * 1.5)
                
                if distance < min_distance:
                    min_distance = distance
                    best_track = track
        
        return best_track
    
    def create_new_track(self, spot, frame_idx):
        """建立新的軌跡"""
        track = {
            'track_id': self.meteor_id_counter,
            'points': [{'x': spot['x'], 'y': spot['y'], 
                       'frame': frame_idx, 'brightness': spot['brightness']}],
            'start_frame': frame_idx,
            'last_update': frame_idx,
            'lost_frames': 0
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
    
    def is_meteor_track(self, track):
        """驗證軌跡是否為流星"""
        points = track['points']
        
        # 至少要有2個點
        if len(points) < 4:
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 點數不足 ({len(points)})")
            return False
        
        # 檢查是否在場景切換附近（前後3幀內）
        start_frame = track['start_frame']
        for scene_frame in self.scene_change_frames:
            if abs(start_frame - scene_frame) <= 3:
                if self.debug:
                    self.log(f"  [DEBUG] Track {track['track_id']}: 在場景切換附近 (frame {start_frame} near {scene_frame})")
                return False
        
        # 計算軌跡長度
        total_length = 0
        for i in range(1, len(points)):
            dx = points[i]['x'] - points[i-1]['x']
            dy = points[i]['y'] - points[i-1]['y']
            total_length += np.sqrt(dx**2 + dy**2)
        
        # 流星應該有足夠長度
        if total_length < 4:
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 軌跡太短 ({total_length:.1f}px)")
            return False
        
        # 計算速度
        duration = points[-1]['frame'] - points[0]['frame']
        if duration == 0:
            duration = 1
        speed = total_length / duration
        
        # 速度限制
        if speed < 2 or speed > 10000:
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 速度超出範圍 ({speed:.1f}px/frame)")
            return False
        
        # 線性度檢查
        if len(points) >= 3:
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
                
                # 線性度要求
                if mean_deviation > 15.0:
                    if self.debug:
                        self.log(f"  [DEBUG] Track {track['track_id']}: 線性度不佳 ({mean_deviation:.1f})")
                    return False
            except:
                pass
        
        # 亮度檢查
        brightnesses = [p['brightness'] for p in points]
        max_brightness = max(brightnesses)
        
        if max_brightness < 2:
            if self.debug:
                self.log(f"  [DEBUG] Track {track['track_id']}: 亮度太低 ({max_brightness:.1f})")
            return False
        
        return True
    
    def merge_overlapping_meteors(self):
        """合併重疊的流星軌跡"""
        if len(self.meteors) <= 1:
            return
        
        merged = []
        used = set()
        
        for i, meteor1 in enumerate(self.meteors):
            if i in used:
                continue
            
            # 收集所有與meteor1重疊的流星
            group = [meteor1]
            group_indices = {i}
            
            for j, meteor2 in enumerate(self.meteors):
                if j <= i or j in used:
                    continue
                
                # 檢查時間重疊
                r1 = meteor1['frame_range']
                r2 = meteor2['frame_range']
                time_overlap = not (r1[1] < r2[0] or r2[1] < r1[0])
                
                if not time_overlap:
                    continue
                
                # 檢查空間重疊
                traj1 = meteor1['trajectory']
                traj2 = meteor2['trajectory']
                
                min_dist = float('inf')
                for p1 in traj1:
                    for p2 in traj2:
                        dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                        min_dist = min(min_dist, dist)
                
                # 如果距離很近，認為是同一顆流星
                if min_dist < 10:
                    group.append(meteor2)
                    group_indices.add(j)
            
            # 合併這組流星
            if len(group) > 2:
                merged_meteor = self.merge_meteor_group(group)
                merged.append(merged_meteor)
                used.update(group_indices)
                if self.debug:
                    self.log(f"  [合併] {len(group)} 個軌跡合併為 {merged_meteor['meteor_id']}")
            else:
                merged.append(meteor1)
                used.add(i)
        
        self.meteors = merged
    
    def merge_meteor_group(self, group):
        """合併一組流星軌跡"""
        # 合併所有軌跡點
        all_points = []
        for meteor in group:
            all_points.extend(meteor['trajectory'])
        
        # 按幀號排序
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
        
        # 計算新的參數
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
            cv2.putText(vis, f"{spot['brightness']:.0f}", 
                       (spot['x']+10, spot['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 組合圖像
        combined = np.hstack([vis, delta_color])
        
        # 加入資訊
        scene_text = " [場景切換]" if is_scene_change else ""
        info = f"Frame {frame_idx} | Spots: {len(spots)} | Active: {len(self.active_tracks)}{scene_text}"
        cv2.putText(combined, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255) if not is_scene_change else (0, 0, 255), 2)
        
        filename = self.debug_folder / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(filename), combined)
    
    def process_video(self):
        """處理整個影片"""
        self.log("="*80)
        self.log("流星偵測系統 - Delta Frame 優化版")
        self.log("="*80)
        self.log(f"影片: {self.video_path}")
        self.log(f"總幀數: {self.total_frames}, FPS: {self.fps:.2f}")
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
        lost_track_threshold = 5
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0)
            
            # 計算幀差分
            delta = cv2.absdiff(curr_gray, prev_gray)
            
            # 檢查場景切換
            is_scene_change = self.is_scene_change(delta)
            if is_scene_change:
                self.scene_change_frames.append(frame_idx)
                if self.debug:
                    self.log(f"  [場景切換] Frame {frame_idx}")
                # 清空所有活躍軌跡
                self.active_tracks.clear()
                prev_gray = curr_gray
                frame_idx += 1
                continue
            
            # 找出明亮的移動點
            spots = self.find_bright_spots(delta, threshold=15)
            
            # 調試輸出
            if self.debug and len(spots) > 0:
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
                    self.log(f"✓ 偵測到流星 #{len(self.meteors)}: "
                           f"幀 {meteor['frame_range'][0]}-{meteor['frame_range'][1]} "
                           f"(時間 {meteor['frame_range'][0]/self.fps:.1f}s-{meteor['frame_range'][1]/self.fps:.1f}s), "
                           f"長度 {meteor['length']:.1f}px, "
                           f"速度 {meteor['speed']:.1f}px/frame")
            
            prev_gray = curr_gray
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                self.log(f"處理進度: {frame_idx}/{self.total_frames}, "
                       f"活躍軌跡: {len(self.active_tracks)}")
        
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
    parser = argparse.ArgumentParser(description='流星偵測 - Delta Frame 優化版本')
    parser.add_argument('--video', type=str, required=True, help='影片路徑')
    parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ 找不到影片: {args.video}")
        return
    
    output_folder = Path("output_delta") / Path(args.video).stem
    tracker = MeteorTracker(args.video, output_folder, debug=args.debug)
    tracker.process_video()

if __name__ == "__main__":
    main()