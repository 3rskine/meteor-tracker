"""
流星偵測系統 - V4.0 密度過濾與動態評分版
針對痛點：
1. [雲層殺手] 引入「密度過濾」：雲層雜訊通常成群出現，流星通常孤立。密集區自動提高審核標準。
2. [拯救短流星] 引入「動態長寬比」：允許短流星長寬比低一點，但直線度要高。
3. [數據可視化] 直接在圖上印出 L(直線度), A(長寬比), N(鄰居數)，不再瞎猜。
"""
import cv2
import numpy as np
import csv
from pathlib import Path
import argparse
from datetime import datetime

class MeteorDetector:
    def __init__(self, 
                 min_length=10,       
                 hough_threshold=4,   
                 canny_low=10,        
                 canny_high=60,
                 max_line_gap=20,
                 min_fill_ratio=0.5): # [放寬] 降低填充率要求，避免斷裂流星被殺
        
        self.min_length = min_length
        self.hough_threshold = hough_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.max_line_gap = max_line_gap
        self.min_fill_ratio = min_fill_ratio
    
    def preprocess_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Median Blur 去雲
        bg = cv2.medianBlur(gray, 21)
        diff = cv2.absdiff(gray, bg)
        
        # 2. Sigma Clipping 低門檻 (0.5 std) - 廣撒網
        mean, std = cv2.meanStdDev(diff)
        thresh_val = mean[0][0] + 0.5 * std[0][0] 
        _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        # 3. 基礎過濾 (只濾掉 1px 噪點)
        mask = np.zeros_like(gray)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 2: 
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        return gray, mask

    def check_line_fill_ratio(self, binary_image, x1, y1, x2, y2):
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length == 0: return 0
        x_coords = np.linspace(x1, x2, length).astype(int)
        y_coords = np.linspace(y1, y2, length).astype(int)
        h, w = binary_image.shape
        valid = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        if len(x_coords) == 0: return 0
        pixels = binary_image[y_coords, x_coords]
        return np.count_nonzero(pixels) / len(pixels)

    def check_linearity(self, contour):
        """計算直線度誤差 (越小越直)"""
        points = contour.reshape(-1, 2)
        if len(points) < 5: return 0.0 # 點太少視為直線
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        error_sum = 0
        for p in points:
            px, py = p
            dist = abs(-vy[0]*(px-x0[0]) + vx[0]*(py-y0[0]))
            error_sum += dist
        return error_sum / len(points)

    def detect_pipeline(self, image, osd_bbox):
        gray, binary = self.preprocess_image(image)
        
        # OSD 遮罩
        h, w = gray.shape
        mask_osd = np.ones((h, w), dtype=np.uint8) * 255
        x, y, bw, bh = osd_bbox
        x = max(0, x); y = max(0, y)
        bw = min(bw, w-x); bh = min(bh, h-y)
        mask_osd[y:y+bh, x:x+bw] = 0
        masked_binary = cv2.bitwise_and(binary, binary, mask=mask_osd)

        # Step 1: Hough 抓線
        lines = cv2.HoughLinesP(masked_binary, 1, np.pi/180, self.hough_threshold, 
                                minLineLength=self.min_length, maxLineGap=self.max_line_gap)
        
        fusion_mask = np.zeros_like(gray)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                fill_ratio = self.check_line_fill_ratio(masked_binary, x1, y1, x2, y2)
                # 這裡放寬一點，讓更多線進入融合階段
                if fill_ratio >= self.min_fill_ratio:
                    cv2.line(fusion_mask, (x1, y1), (x2, y2), 255, thickness=4)

        # Step 2: 視覺融合
        kernel_fuse = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fusion_mask = cv2.dilate(fusion_mask, kernel_fuse, iterations=1)
        contours, _ = cv2.findContours(fusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 20: continue
            
            rect = cv2.minAreaRect(cnt)
            (center), (width, height), angle = rect
            if width < height: width, height = height, width
            
            aspect_ratio = width / height if height > 0 else 0
            linearity = self.check_linearity(cnt)
            
            candidates.append({
                'cnt': cnt,
                'box': np.int0(cv2.boxPoints(rect)),
                'center': center,
                'w': width,
                'h': height,
                'ar': aspect_ratio,
                'lin': linearity,
                'status': 'UNKNOWN' # 待定
            })

        # Step 3: [新功能] 密度計算 (Density Check)
        # 計算每個候選框附近 100px 內有多少其他候選框
        for i, c1 in enumerate(candidates):
            neighbors = 0
            for j, c2 in enumerate(candidates):
                if i == j: continue
                dist = np.linalg.norm(np.array(c1['center']) - np.array(c2['center']))
                if dist < 100: # 100px 內的鄰居
                    neighbors += 1
            c1['neighbors'] = neighbors

        # Step 4: [新功能] 動態評分過濾 (Dynamic Scoring)
        final_meteors = []
        rejected = []

        for c in candidates:
            is_meteor = False
            reason = ""

            # A. 密度檢查：如果鄰居太多 (>2)，判定為雲層群聚
            # 但如果它「極度像流星」(又直又長)，我們可以豁免它 (例如流星雨)
            is_crowded = c['neighbors'] > 2
            
            # B. 動態長寬比
            # 如果物體很長 (>50px)，長寬比要求正常 (>2.0)
            # 如果物體很短 (<50px)，允許胖一點 (AR > 1.3)，因為 fusion 讓它變胖了
            ar_threshold = 2.0 if c['w'] > 50 else 1.3
            
            # C. 直線度要求
            # 如果是密集區(雲)，直線度要求極高 (<1.0)
            # 如果是孤立區，直線度要求正常 (<2.0)
            lin_threshold = 1.0 if is_crowded else 2.0

            # --- 綜合判斷 ---
            if c['ar'] > ar_threshold:
                if c['lin'] < lin_threshold:
                    is_meteor = True
                else:
                    reason = "彎曲"
            else:
                reason = "太圓"

            # 強制復活：如果它非常長且非常直，就算在雲裡也要抓
            if c['w'] > 80 and c['lin'] < 1.0:
                is_meteor = True

            # 記錄結果
            c['status'] = 'METEOR' if is_meteor else f'REJECT({reason})'
            
            if is_meteor:
                final_meteors.append(c)
            else:
                rejected.append(c)

        return {
            'final': final_meteors,
            'rejected': rejected,
            'debug_mask': fusion_mask
        }

class ImageProcessor:
    def __init__(self, detector, base_output_folder, annotated_folder, debug_folder, debug=False):
        self.detector = detector
        self.base_output_folder = Path(base_output_folder)
        self.annotated_folder = Path(annotated_folder)
        self.debug_folder = Path(debug_folder)
        self.debug = debug
        self.log_file = self.base_output_folder / "process_log.txt"
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def process(self, image_path, osd_bbox):
        image = cv2.imread(str(image_path))
        if image is None: return None
        
        start_time = datetime.now()
        results = self.detector.detect_pipeline(image, osd_bbox)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 只要有最終判定為流星的才算 detected (或者你可以改成有 candidate 就算)
        detected = len(results['final']) > 0 
        
        if detected:
            self.log(f"✓ {image_path.name}: 偵測到 {len(results['final'])} 個流星")
        else:
            print(".", end="", flush=True)
            
        self._create_visualization(image, results, osd_bbox, image_path.name)
            
        return {
            'file_name': image_path.name,
            'processing_time': processing_time,
            'meteor_count': len(results['final']),
            'detected': detected
        }
    
    def _create_visualization(self, image, results, osd_bbox, original_filename):
        vis_img = image.copy()
        
        # 1. 畫出被拒絕的 (紅色虛線框)
        for c in results['rejected']:
            box = c['box']
            cv2.drawContours(vis_img, [box], 0, (0, 0, 255), 1)
            # [數據顯示] 在框旁印出數據: L=Linearity, A=Aspect Ratio, N=Neighbors
            label = f"L:{c['lin']:.1f} A:{c['ar']:.1f} N:{c['neighbors']}"
            cv2.putText(vis_img, label, tuple(box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 2. 畫出最終流星 (黃色實線框)
        for c in results['final']:
            box = c['box']
            cv2.drawContours(vis_img, [box], 0, (0, 255, 255), 2)
            label = f"L:{c['lin']:.1f} A:{c['ar']:.1f} N:{c['neighbors']}"
            cv2.putText(vis_img, label, tuple(box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # OSD
        x, y, w, h = osd_bbox
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (100, 100, 100), 1)
        
        # 存圖 (這裡強制全部存，讓你分析)
        cv2.imwrite(str(self.annotated_folder / f"{Path(original_filename).stem}_result.jpg"), vis_img)
        
        if self.debug:
            cv2.imwrite(str(self.debug_folder / f"{Path(original_filename).stem}_mask.jpg"), results['debug_mask'])

class BatchProcessor:
    def __init__(self, detector, input_folder, output_base_folder, debug=False):
        self.detector = detector
        self.input_folder = Path(input_folder)
        self.output_base_folder = Path(output_base_folder)
        self.debug = debug
        self.output_base_folder.mkdir(parents=True, exist_ok=True)
        self.annotated_folder = self.output_base_folder / "annotated_images"
        self.annotated_folder.mkdir(exist_ok=True)
        self.debug_folder = self.output_base_folder / "debug_data"
        self.debug_folder.mkdir(exist_ok=True)
        self.results = []
        
    def process_all(self, osd_bbox):
        print("=" * 80)
        print("流星偵測系統 - V4.0 密度過濾與動態評分版")
        print("顯示數據：L=直線度(越小越直), A=長寬比, N=100px內鄰居數")
        print("=" * 80)
        images = sorted(self.input_folder.glob('*.[jJ][pP]*[gG]'))
        if not images: return []
        processor = ImageProcessor(self.detector, self.output_base_folder, self.annotated_folder, self.debug_folder, self.debug)
        for image_path in images:
            result = processor.process(image_path, osd_bbox)
            if result: self.results.append(result)
        self._generate_summary()
    
    def _generate_summary(self):
        total = len(self.results)
        detected = sum(1 for r in self.results if r['detected'])
        print(f"\n\n處理完成: {detected}/{total} 張偵測到流星 ({detected/total*100:.1f}%)")
        csv_path = self.output_base_folder / "final_report.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['檔名', '偵測到', '數量', '耗時'])
            for r in self.results:
                writer.writerow([r['file_name'], 'Yes' if r['detected'] else 'No', r['meteor_count'], f"{r['processing_time']:.3f}"])

def parse_osd_bbox(arg):
    try: return tuple(map(int, arg.split(',')))
    except: return (0, 460, 495, 80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_result')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--osd-bbox', type=parse_osd_bbox, default=(0, 460, 495, 80))
    args = parser.parse_args()
    
    detector = MeteorDetector()
    BatchProcessor(detector, args.folder, args.output, args.debug).process_all(args.osd_bbox)

if __name__ == '__main__':
    main()