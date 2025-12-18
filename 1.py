"""
流星偵測系統 - V3 形狀過濾版
針對問題：
1. 保留短流星偵測能力 (靈敏度)。
2. 消除雲層邊緣產生的破碎線條雜訊 (特異性)。
核心技術：
- 使用 Median Blur 進行背景估計 (比 TopHat 更適合雲層)。
- 加入「連通區域分析 (Connected Components)」與「長寬比 (Aspect Ratio)」過濾。
"""
import cv2
import numpy as np
import csv
from pathlib import Path
import argparse
from datetime import datetime

class MeteorDetector:
    def __init__(self, 
                 min_length=5,        # 保持極短流星偵測
                 min_brightness=10, 
                 min_angle=5,
                 hough_threshold=8,   # [降低] 讓Hough更容易抓到線，因為我們前面會先把雜訊濾得很乾淨
                 canny_low=30,        # [降低] 允許微弱邊緣，靠幾何形狀來過濾雜訊
                 canny_high=100,
                 max_line_gap=10):
        
        self.min_length = min_length
        self.min_brightness = min_brightness
        self.min_angle = min_angle
        self.hough_threshold = hough_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.max_line_gap = max_line_gap
    
    def preprocess_image(self, image):
        """
        V3 預處理流程 (修正版)：
        1. 背景減除 (Median Blur)
        2. 自適應閾值 (Adaptive Threshold) -> 修正了型態錯誤
        3. 形狀過濾 (Shape Filtering)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # --- 步驟 1: 背景減除 ---
        bg = cv2.medianBlur(gray, 21)
        diff = cv2.absdiff(gray, bg)
        
        # --- 步驟 2: 自適應閾值 (Sigma Clipping) ---
        mean, std = cv2.meanStdDev(diff)
        
        # [修正點] mean 和 std 是 numpy 陣列，必須取出裡面的純量數值 (scalar)
        # 例如從 [[3.5]] 變成 3.5
        thresh_val = mean[0][0] + 3.0 * std[0][0] 
        
        _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        # --- 步驟 3: 形狀過濾 ---
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3: continue 
            
            rect = cv2.minAreaRect(cnt)
            (center), (width, height), angle = rect
            
            if width < height:
                width, height = height, width
            
            if height == 0: 
                aspect_ratio = 999 
            else:
                aspect_ratio = width / height
            
            if aspect_ratio > 1.5:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
            elif area > 20 and aspect_ratio > 1.2:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        final_processed = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        return gray, final_processed


    def apply_osd_mask(self, image, osd_bbox):
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        x, y, bbox_w, bbox_h = osd_bbox
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        x_end = min(x + bbox_w, w)
        y_end = min(y + bbox_h, h)
        mask[y:y_end, x:x_end] = 0
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked, mask
    
    def detect_lines(self, processed_image):
        """偵測直線"""
        # 因為我們在 preprocess 已經做過嚴格的形狀過濾，這裡的 Canny 可以很單純
        edges = cv2.Canny(processed_image, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.hough_threshold,
            minLineLength=self.min_length,
            maxLineGap=self.max_line_gap
        )
        return lines, edges
    
    def calculate_line_brightness(self, gray_image, x1, y1, x2, y2):
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length < 2: return 0
        xs = np.linspace(x1, x2, length).astype(int)
        ys = np.linspace(y1, y2, length).astype(int)
        h, w = gray_image.shape
        valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not valid_mask.any(): return 0
        
        brightnesses = gray_image[ys[valid_mask], xs[valid_mask]]
        return float(np.percentile(brightnesses, 85)) # 取前 15% 亮的平均
    
    def filter_meteor_candidates(self, lines, gray_image, osd_bbox):
        if lines is None: return []
        meteors = []
        x_osd, y_osd, w_osd, h_osd = osd_bbox
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            
            if length < self.min_length: continue
            
            angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
            if angle < self.min_angle: continue
            
            mid_y = (y1 + y2) // 2
            if y_osd <= mid_y <= y_osd + h_osd: continue
            
            line_brightness = self.calculate_line_brightness(gray_image, x1, y1, x2, y2)
            if line_brightness < self.min_brightness: continue
            
            meteor = {
                'start_point': [int(x1), int(y1)],
                'end_point': [int(x2), int(y2)],
                'length': float(length),
                'angle': float(angle),
                'brightness': float(line_brightness)
            }
            meteors.append(meteor)
        return meteors
    
    def detect(self, image, osd_bbox=(0, 460, 470, 80)):
        gray, processed_image = self.preprocess_image(image)
        masked_processed, mask = self.apply_osd_mask(processed_image, osd_bbox)
        lines, edges = self.detect_lines(masked_processed)
        meteors = self.filter_meteor_candidates(lines, gray, osd_bbox)
        
        debug_info = {
            'edges': edges,
            'mask': mask,
            'processed': processed_image, 
            'raw_lines': len(lines) if lines is not None else 0
        }
        return meteors, debug_info


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
    
    def process(self, image_path, osd_bbox=(0, 460, 470, 80)):
        image = cv2.imread(str(image_path))
        if image is None: return None
        
        start_time = datetime.now()
        meteors, debug_info = self.detector.detect(image, osd_bbox)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        detected = len(meteors) > 0
        
        if detected:
            self.log(f"✓ {image_path.name}: {len(meteors)} 條軌跡")
            self._create_annotated_image(image, meteors, osd_bbox, image_path.name)
        else:
            print(".", end="", flush=True)
            
        result = {
            'file_name': image_path.name,
            'processing_time': processing_time,
            'meteor_count': len(meteors),
            'detected': detected,
            'meteors': meteors
        }
        
        if self.debug:
            self._save_debug_data(image_path, result, debug_info)
            
        return result
    
    def _create_annotated_image(self, image, meteors, osd_bbox, original_filename):
        annotated = image.copy()
        for meteor in meteors:
            x1, y1 = meteor['start_point']
            x2, y2 = meteor['end_point']
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # 對於極短流星，畫個框框標示，比較容易看清楚
            if meteor['length'] < 15:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                cv2.rectangle(annotated, (cx-10, cy-10), (cx+10, cy+10), (0, 255, 255), 1)

        x, y, w, h = osd_bbox
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.imwrite(str(self.annotated_folder / f"{Path(original_filename).stem}_marked.jpg"), annotated)

    def _save_debug_data(self, image_path, result, debug_info):
        cv2.imwrite(str(self.debug_folder / f"{Path(image_path.name).stem}_processed.jpg"), debug_info['processed'])
        cv2.imwrite(str(self.debug_folder / f"{Path(image_path.name).stem}_edges.jpg"), debug_info['edges'])


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
        
    def process_all(self, osd_bbox=(0, 460, 470, 80)):
        print("=" * 80)
        print("流星偵測系統 - V3 形狀過濾版啟動")
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
    except: return (0, 460, 470, 80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_result')
    parser.add_argument('--min-length', type=int, default=5) 
    parser.add_argument('--min-brightness', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--osd-bbox', type=parse_osd_bbox, default=(0, 460, 470, 80))
    args = parser.parse_args()
    
    detector = MeteorDetector(min_length=args.min_length, min_brightness=args.min_brightness)
    BatchProcessor(detector, args.folder, args.output, args.debug).process_all(args.osd_bbox)

if __name__ == '__main__':
    main()