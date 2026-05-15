"""
医学细胞分类与计数系统 - 推理模块
实现细胞检测、分类和计数功能
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import time


class CellDetector:
    """细胞检测器"""
    
    def __init__(self, model_path, class_names=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化检测器
        
        Args:
            model_path: 模型权重文件路径
            class_names: 类别名称字典或列表
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        if class_names is None:
            self.class_names = {
                0: 'Platelets',
                1: 'RBC',
                2: 'WBC'
            }
        elif isinstance(class_names, list):
            self.class_names = {i: name for i, name in enumerate(class_names)}
        else:
            self.class_names = class_names
        
        self.class_colors = self._generate_colors()
    
    def _generate_colors(self):
        """为每个类别生成颜色"""
        np.random.seed(42)
        colors = {}
        for cls_id in self.class_names:
            colors[cls_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return colors
    
    def detect(self, image, conf_threshold=None, iou_threshold=None):
        """
        检测细胞
        
        Args:
            image: 输入图像（路径或 numpy 数组）
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            
        Returns:
            检测结果
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        return results
    
    def detect_and_count(self, image, conf_threshold=None):
        """
        检测并计数细胞
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果和计数统计
        """
        results = self.detect(image, conf_threshold)
        
        count_stats = defaultdict(int)
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                conf = confs[i]
                cls = int(classes[i])
                
                count_stats[cls] += 1
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': cls,
                    'class_name': self.class_names.get(cls, 'Unknown')
                })
        
        return {
            'detections': detections,
            'count_stats': dict(count_stats),
            'total_count': sum(count_stats.values()),
            'image_shape': result.orig_shape
        }
    
    def draw_detections(self, image, results, show_labels=True, show_conf=True):
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            results: 检测结果
            show_labels: 是否显示标签
            show_conf: 是否显示置信度
            
        Returns:
            绘制后的图像
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        img = image.copy()
        
        if isinstance(results, list):
            for result in results:
                self._draw_single_result(img, result, show_labels, show_conf)
        else:
            self._draw_single_result(img, results, show_labels, show_conf)
        
        return img
    
    def _draw_single_result(self, img, result, show_labels, show_conf):
        """绘制单个检测结果"""
        if result.boxes is None:
            return
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = confs[i]
            cls = int(classes[i])
            
            color = self.class_colors.get(cls, (0, 255, 0))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            if show_labels:
                label = self.class_names.get(cls, f'Class {cls}')
                if show_conf:
                    label += f' {conf:.2f}'
                
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(
                    img,
                    (x1, y1 - text_h - baseline - 5),
                    (x1 + text_w, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
    
    def save_results(self, image, output_path, results):
        """
        保存检测结果图像
        
        Args:
            image: 输入图像或路径
            output_path: 输出路径
            results: 检测结果
        """
        result_img = self.draw_detections(image, results)
        cv2.imwrite(str(output_path), result_img)
        print(f"结果已保存到：{output_path}")
    
    def batch_detect(self, image_dir, output_dir=None, conf_threshold=None):
        """
        批量检测
        
        Args:
            image_dir: 图像目录
            output_dir: 输出目录
            conf_threshold: 置信度阈值
            
        Returns:
            所有检测结果的统计
        """
        image_dir = Path(image_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_stats = []
        total_stats = defaultdict(int)
        
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue
            
            print(f"处理：{img_path.name}")
            
            result = self.detect_and_count(str(img_path), conf_threshold)
            all_stats.append({
                'filename': img_path.name,
                'stats': result
            })
            
            for cls, count in result['count_stats'].items():
                total_stats[cls] += count
            
            if output_dir:
                results = self.detect(str(img_path), conf_threshold)
                self.save_results(
                    str(img_path),
                    output_dir / img_path.name,
                    results
                )
        
        return {
            'per_image_stats': all_stats,
            'total_stats': dict(total_stats),
            'total_count': sum(total_stats.values()),
            'image_count': len(all_stats)
        }
    
    def detect_video(self, video_path, output_path=None, conf_threshold=None):
        """
        检测视频中的细胞
        
        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            conf_threshold: 置信度阈值
            
        Returns:
            检测统计
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        frame_count = 0
        all_stats = []
        
        print(f"开始处理视频，共 {total_frames} 帧...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"处理进度：{frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            result = self.detect_and_count(frame, conf_threshold)
            all_stats.append(result)
            
            result_img = self.draw_detections(frame, self.detect(frame, conf_threshold))
            
            cv2.putText(
                result_img,
                f"Frame: {frame_count}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                result_img,
                f"Total: {result['total_count']}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            if out:
                out.write(result_img)
            
            if out is None and frame_count <= 10:
                cv2.imshow('Detection', result_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if out:
            out.release()
        
        if output_path:
            print(f"结果视频已保存到：{output_path}")
        
        return {
            'frame_count': frame_count,
            'per_frame_stats': all_stats,
            'video_path': output_path
        }


class CellCounter:
    """细胞计数器 - 提供高级计数功能"""
    
    def __init__(self, detector):
        """
        初始化计数器
        
        Args:
            detector: CellDetector 实例
        """
        self.detector = detector
    
    def count_by_region(self, image, regions, conf_threshold=None):
        """
        按区域计数细胞
        
        Args:
            image: 输入图像
            regions: 区域字典 {'region_name': [x1, y1, x2, y2]}
            conf_threshold: 置信度阈值
            
        Returns:
            各区域的计数统计
        """
        result = self.detector.detect_and_count(image, conf_threshold)
        detections = result['detections']
        
        region_stats = {}
        for region_name, region_bbox in regions.items():
            region_stats[region_name] = defaultdict(int)
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                rx1, ry1, rx2, ry2 = region_bbox
                
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    region_stats[region_name][det['class']] += 1
        
        return {
            'region_stats': {k: dict(v) for k, v in region_stats.items()},
            'total_detections': len(detections)
        }
    
    def generate_report(self, detection_result, output_path='report.txt'):
        """
        生成检测报告
        
        Args:
            detection_result: 检测结果
            output_path: 输出文件路径
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("医学细胞检测报告")
        report_lines.append("=" * 60)
        report_lines.append(f"报告生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if 'count_stats' in detection_result:
            report_lines.append("【细胞统计】")
            for cls_id, count in detection_result['count_stats'].items():
                class_name = self.detector.class_names.get(cls_id, f'Class {cls_id}')
                report_lines.append(f"  {class_name}: {count}")
            
            report_lines.append(f"\n总计：{detection_result['total_count']}")
        
        if 'image_shape' in detection_result:
            h, w = detection_result['image_shape'][:2]
            report_lines.append(f"\n图像尺寸：{w} x {h}")
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"报告已保存到：{output_path}")
        return report_text


def quick_detect(image_path, model_path, class_names=None, conf=0.25):
    """
    快速检测函数
    
    Args:
        image_path: 图像路径
        model_path: 模型路径
        class_names: 类别名称
        conf: 置信度阈值
        
    Returns:
        检测结果
    """
    detector = CellDetector(model_path, class_names, conf_threshold=conf)
    result = detector.detect_and_count(image_path)
    
    print("\n检测结果:")
    for cls_id, count in result['count_stats'].items():
        class_name = detector.class_names.get(cls_id, f'Class {cls_id}')
        print(f"  {class_name}: {count}")
    print(f"总计：{result['total_count']}")
    
    return result


if __name__ == '__main__':
    detector = CellDetector(
        model_path=r"D:\code\demo\ultralytics-8.3.163\runs\segment\train\weights\best.pt",
        class_names=['Platelets', 'RBC', 'WBC'],
        conf_threshold=0.25
    )
    
    result = detector.detect_and_count(
        r"D:\code\demo\ultralytics-8.3.163\datasets\images\val\BloodImage_00007.jpg"
    )
    
    print("\n检测结果:")
    for cls_id, count in result['count_stats'].items():
        print(f"  {detector.class_names[cls_id]}: {count}")
    print(f"总计：{result['total_count']}")
