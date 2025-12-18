"""
流星偵測系統 - 高敏感度調整版
基於 V1 架構修改，專注於提升偵測率（Recall Rate）
修改重點：
1. 允許線段斷裂重連 (max_line_gap)
2. 降低 Canny 與 Hough 門檻
3. 改用中位數計算亮度，避免被背景拉低
"""
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime

class MeteorDetector:
    """流星偵測核心引擎"""
    
    def __init__(self, 
                 min_length=10,       # [修改] 預設從 20 降到 10
                 min_brightness=40,   # [修改] 預設從 80 降到 40
                 min_angle=5,
                 hough_threshold=15,  # [修改] 預設從 30 降到 15
                 canny_low=30,
                 canny_high=60,       # [修改] 預設從 100 降到 60 (抓暗線)
                 max_line_gap=25):    # [新增] 允許斷線的最大距離
        """
        初始化偵測參數
        """
        self.min_length = min_length
        self.min_brightness = min_brightness
        self.min_angle = min_angle
        self.hough_threshold = hough_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.max_line_gap = max_line_gap
    
    def preprocess_image(self, image):
        """圖片預處理"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # [修改] 高斯模糊從 (5, 5) 改為 (3, 3)，保留更多暗流星細節
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray, blurred
    
    def apply_osd_mask(self, image, osd_bbox):
        """套用 OSD 遮罩"""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        x, y, bbox_w, bbox_h = osd_bbox
        
        # 確保不超出邊界
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        x_end = min(x + bbox_w, w)
        y_end = min(y + bbox_h, h)
        
        mask[y:y_end, x:x_end] = 0
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        return masked, mask
    
    def detect_lines(self, image):
        """使用 Hough Transform 偵測直線"""
        edges = cv2.Canny(image, self.canny_low, self.canny_high, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.hough_threshold,
            minLineLength=self.min_length,
            maxLineGap=self.max_line_gap  # [修改] 使用變數控制斷線連接
        )
        
        return lines, edges
    
    def calculate_line_brightness(self, gray_image, x1, y1, x2, y2):
        """計算線段沿線的亮度"""
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length < 2:
            return 0
        
        xs = np.linspace(x1, x2, length).astype(int)
        ys = np.linspace(y1, y2, length).astype(int)
        
        h, w = gray_image.shape
        valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        
        if not valid_mask.any():
            return 0
        
        brightnesses = gray_image[ys[valid_mask], xs[valid_mask]]
        
        # [修改] 改用中位數 (median)，比平均值更能抵抗背景雜訊
        return float(np.median(brightnesses))
    
    def filter_meteor_candidates(self, lines, gray_image, osd_bbox):
        """過濾流星候選軌跡"""
        if lines is None:
            return []
        
        meteors = []
        x_osd, y_osd, w_osd, h_osd = osd_bbox
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 1. 長度檢查
            length = np.hypot(x2 - x1, y2 - y1)
            if length < self.min_length:
                continue
            
            # 2. 角度檢查（排除水平線）
            angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
            if angle < self.min_angle:
                continue
            
            # 3. OSD 區域檢查
            mid_y = (y1 + y2) // 2
            if y_osd <= mid_y <= y_osd + h_osd:
                continue
            
            # 4. 亮度檢查
            brightness = self.calculate_line_brightness(gray_image, x1, y1, x2, y2)
            if brightness < self.min_brightness:
                continue
            
            # 通過所有過濾
            meteor = {
                'meteor_id': f"meteor_{len(meteors):03d}",
                'start_point': [int(x1), int(y1)],
                'end_point': [int(x2), int(y2)],
                'length': float(length),
                'angle': float(angle),
                'brightness': float(brightness),
                'bbox': [
                    int(min(x1, x2) - 10),
                    int(min(y1, y2) - 10),
                    int(abs(x2 - x1) + 20),
                    int(abs(y2 - y1) + 20)
                ]
            }
            meteors.append(meteor)
        
        return meteors
    
    def detect(self, image, osd_bbox=(0, 460, 470, 80)):
        """主要偵測函數"""
        # 預處理
        gray, blurred = self.preprocess_image(image)
        
        # 套用 OSD 遮罩
        masked, mask = self.apply_osd_mask(blurred, osd_bbox)
        
        # 偵測直線
        lines, edges = self.detect_lines(masked)
        
        # 過濾流星候選
        meteors = self.filter_meteor_candidates(lines, gray, osd_bbox)
        
        debug_info = {
            'edges': edges,
            'mask': mask,
            'raw_lines': len(lines) if lines is not None else 0
        }
        
        return meteors, debug_info


class ImageProcessor:
    """單張圖片處理器"""
    
    def __init__(self, detector, output_folder, debug=False):
        self.detector = detector
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        
        self.log_file = self.output_folder / "detection.log"
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"偵測開始: {datetime.now()}\n")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def process(self, image_path, osd_bbox=(0, 460, 470, 80)):
        """處理單張圖片"""
        self.log("=" * 80)
        self.log(f"處理圖片: {image_path}")
        
        # 讀取圖片
        image = cv2.imread(str(image_path))
        if image is None:
            self.log("❌ 無法讀取圖片")
            return None
        
        h, w = image.shape[:2]
        self.log(f"圖片尺寸: {w}×{h}")
        
        # 偵測
        start_time = datetime.now()
        meteors, debug_info = self.detector.detect(image, osd_bbox)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        self.log(f"偵測到 {len(meteors)} 條軌跡")
        self.log(f"處理時間: {processing_time:.3f} 秒")
        
        # 儲存結果
        result = {
            'image': str(image_path),
            'size': [w, h],
            'osd_region': list(osd_bbox),
            'processing_time': processing_time,
            'meteor_count': len(meteors),
            'raw_lines_detected': debug_info['raw_lines'],
            'meteors': meteors
        }
        
        # 儲存 JSON
        json_path = self.output_folder / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 生成標記圖片
        if meteors:
            self._create_annotated_image(image, meteors, osd_bbox)
        
        # Debug 模式：儲存邊緣檢測圖
        if self.debug:
            cv2.imwrite(
                str(self.output_folder / "edges.jpg"),
                debug_info['edges']
            )
            
            # 儲存遮罩視覺化
            mask_vis = cv2.cvtColor(debug_info['mask'], cv2.COLOR_GRAY2BGR)
            cv2.rectangle(mask_vis, 
                          (osd_bbox[0], osd_bbox[1]),
                          (osd_bbox[0]+osd_bbox[2], osd_bbox[1]+osd_bbox[3]),
                          (0, 0, 255), 2)
            cv2.imwrite(
                str(self.output_folder / "mask.jpg"),
                mask_vis
            )
        
        self.log(f"✓ 結果已儲存至: {self.output_folder}")
        
        return result
    
    def _create_annotated_image(self, image, meteors, osd_bbox):
        """生成標記圖片"""
        annotated = image.copy()
        
        # 畫流星軌跡
        for meteor in meteors:
            x1, y1 = meteor['start_point']
            x2, y2 = meteor['end_point']
            
            # 軌跡線（綠色）
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 起點（紅色）
            cv2.circle(annotated, (x1, y1), 6, (0, 0, 255), -1)
            
            # 終點（藍色）
            cv2.circle(annotated, (x2, y2), 6, (255, 0, 0), -1)
            
            # 標籤
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            label = f"{meteor['meteor_id']} ({meteor['length']:.0f}px)"
            cv2.putText(annotated, label,
                        (mid_x, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 畫 OSD 區域
        x, y, w, h = osd_bbox
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(annotated, "OSD MASKED", (x+10, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 總結資訊
        info = f"Meteors: {len(meteors)}"
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 儲存
        output_path = self.output_folder / "annotated.jpg"
        cv2.imwrite(str(output_path), annotated)
        self.log(f"✓ 標記圖片: {output_path}")


class BatchProcessor:
    """批次處理器"""
    
    def __init__(self, detector, input_folder, output_base_folder, debug=False):
        self.detector = detector
        self.input_folder = Path(input_folder)
        self.output_base_folder = Path(output_base_folder)
        self.output_base_folder.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        
        self.results = []
        self.log_file = self.output_base_folder / "batch_processing.log"
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"批次處理開始: {datetime.now()}\n")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def find_images(self):
        """尋找所有圖片"""
        extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(self.input_folder.glob(ext))
        return sorted(set(images))
    
    def process_all(self, osd_bbox=(0, 460, 470, 80)):
        """批次處理所有圖片"""
        self.log("=" * 80)
        self.log("批次流星偵測（高敏感調整版）")
        self.log("=" * 80)
        self.log(f"輸入資料夾: {self.input_folder}")
        self.log(f"輸出資料夾: {self.output_base_folder}")
        self.log(f"OSD 遮罩: {osd_bbox}\n")
        
        images = self.find_images()
        
        if not images:
            self.log("❌ 找不到任何圖片")
            return []
        
        self.log(f"找到 {len(images)} 張圖片\n")
        
        batch_start = datetime.now()
        
        # 處理每張圖片
        for i, image_path in enumerate(images, 1):
            self.log(f"\n處理 [{i}/{len(images)}]: {image_path.name}")
            
            output_folder = self.output_base_folder / image_path.stem
            
            try:
                processor = ImageProcessor(
                    self.detector,
                    output_folder,
                    debug=self.debug
                )
                
                result = processor.process(image_path, osd_bbox)
                
                if result:
                    self.results.append({
                        'file_name': image_path.name,
                        'file_path': str(image_path),
                        'meteor_count': result['meteor_count'],
                        'processing_time': result['processing_time'],
                        'status': 'success'
                    })
                    self.log(f"✓ 成功: {result['meteor_count']} 條軌跡")
                else:
                    self.results.append({
                        'file_name': image_path.name,
                        'file_path': str(image_path),
                        'status': 'failed',
                        'error': 'Processing returned None'
                    })
                    
            except Exception as e:
                self.log(f"❌ 錯誤: {str(e)}")
                self.results.append({
                    'file_name': image_path.name,
                    'file_path': str(image_path),
                    'status': 'failed',
                    'error': str(e)
                })
        
        batch_end = datetime.now()
        total_time = (batch_end - batch_start).total_seconds()
        
        # 生成總結報告
        self._generate_summary(total_time)
        
        return self.results
    
    def _generate_summary(self, total_time):
        """生成總結報告"""
        self.log("\n" + "=" * 80)
        self.log("批次處理完成")
        self.log("=" * 80)
        
        success = sum(1 for r in self.results if r['status'] == 'success')
        failed = len(self.results) - success
        total_meteors = sum(r.get('meteor_count', 0) for r in self.results)
        
        self.log(f"總圖片數: {len(self.results)}")
        self.log(f"成功: {success}, 失敗: {failed}")
        self.log(f"總流星數: {total_meteors}")
        self.log(f"總耗時: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
        
        if success > 0:
            avg_time = total_time / success
            self.log(f"平均每張: {avg_time:.2f} 秒")
        
        # 儲存 JSON 總結
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(self.results),
            'success_count': success,
            'failed_count': failed,
            'total_meteors': total_meteors,
            'total_time_seconds': total_time,
            'results': self.results
        }
        
        summary_path = self.output_base_folder / "batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.log(f"\n✓ 總結報告: {summary_path}")


def parse_osd_bbox(arg):
    """解析 OSD bbox 字串"""
    try:
        parts = [int(p.strip()) for p in arg.split(',')]
        if len(parts) != 4:
            raise ValueError()
        return tuple(parts)
    except:
        raise argparse.ArgumentTypeError(
            "OSD bbox 格式錯誤，應為: x,y,w,h（例如：0,460,470,80）"
        )


def main():
    parser = argparse.ArgumentParser(
        description='流星偵測系統 - 高敏感度調整版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  # 單張圖片
  python script.py --image photo.jpg
  
  # 批次處理資料夾
  python script.py --folder ./images
  
  # 啟用 debug 模式（查看邊緣檢測結果）
  python script.py --folder ./images --debug
        """
    )
    
    # 輸入來源（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                             help='單張圖片路徑')
    input_group.add_argument('--folder', type=str,
                             help='批次處理資料夾')
    
    # [修改] 將預設參數調整為高敏感度
    parser.add_argument('--min-length', type=int, default=10, 
                        help='最小軌跡長度，預設 10 (原 20)')
    parser.add_argument('--min-brightness', type=int, default=40,
                        help='最小亮度，預設 40 (原 80)')
    parser.add_argument('--min-angle', type=int, default=5,
                        help='最小傾斜角度，預設 5')
    parser.add_argument('--max-line-gap', type=int, default=25,
                        help='允許斷線的最大距離，預設 25 (原 5)')
    
    # OSD 設定
    parser.add_argument('--osd-bbox', type=parse_osd_bbox,
                        default=(0, 460, 470, 80),
                        help='OSD 區域 (x,y,w,h)，預設 0,460,470,80')
    
    # 其他選項
    parser.add_argument('--output', type=str, default='output',
                        help='輸出資料夾，預設 output')
    parser.add_argument('--debug', action='store_true',
                        help='啟用 debug 模式（儲存邊緣檢測圖）')
    
    args = parser.parse_args()
    
    # 建立偵測器
    detector = MeteorDetector(
        min_length=args.min_length,
        min_brightness=args.min_brightness,
        min_angle=args.min_angle,
        # 這裡的寫死參數也設為較低的值
        hough_threshold=15,
        canny_high=60,
        max_line_gap=args.max_line_gap
    )
    
    print("\n" + "=" * 80)
    print("流星偵測系統 - 高敏感度調整版")
    print("=" * 80)
    print(f"偵測參數:")
    print(f"  最小長度: {args.min_length} px")
    print(f"  最小亮度: {args.min_brightness}")
    print(f"  最小角度: {args.min_angle}°")
    print(f"  斷線容許: {args.max_line_gap} px")
    print(f"  OSD 區域: {args.osd_bbox}")
    print("=" * 80 + "\n")
    
    # 單張圖片模式
    if args.image:
        processor = ImageProcessor(detector, args.output, args.debug)
        result = processor.process(args.image, args.osd_bbox)
        
        if result and result['meteor_count'] > 0:
            print(f"\n✓ 偵測完成！找到 {result['meteor_count']} 條流星軌跡")
        else:
            print("\n未偵測到流星軌跡")
            if args.debug:
                 print(f"請檢查輸出資料夾中的 edges.jpg")
    
    # 批次處理模式
    elif args.folder:
        batch_processor = BatchProcessor(
            detector,
            args.folder,
            args.output,
            args.debug
        )
        results = batch_processor.process_all(args.osd_bbox)
        
        success = sum(1 for r in results if r['status'] == 'success')
        total_meteors = sum(r.get('meteor_count', 0) for r in results)
        
        print(f"\n✓ 批次處理完成！")
        print(f"  成功處理: {success}/{len(results)} 張")
        print(f"  總流星數: {total_meteors}")


if __name__ == '__main__':
    main()