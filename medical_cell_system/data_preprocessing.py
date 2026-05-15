"""
医学细胞图像数据预处理模块
功能：图像增强、数据清洗、格式转换
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


class CellDataPreprocessor:
    """医学细胞数据预处理器"""
    
    def __init__(self, root_dir, img_size=640):
        """
        初始化预处理器
        
        Args:
            root_dir: 数据集根目录
            img_size: 图像尺寸
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        
    def enhance_image(self, img, method='clahe'):
        """
        图像增强
        
        Args:
            img: 输入图像
            method: 增强方法 ('clahe', 'gamma', 'histogram')
            
        Returns:
            增强后的图像
        """
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        if method == 'clahe':
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l)
            enhanced_lab = cv2.merge((enhanced_l, a, b))
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return enhanced_img
        
        elif method == 'gamma':
            gamma = 1.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** (1.0 / invGamma)) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(img, table)
        
        elif method == 'histogram':
            return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        
        return img
    
    def resize_with_padding(self, img, target_size=640):
        """
        保持宽高比的缩放并填充
        
        Args:
            img: 输入图像
            target_size: 目标尺寸
            
        Returns:
            处理后的图像和缩放信息
        """
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return padded, scale, (left, top)
    
    def augment_dataset(self, source_dir, target_dir, augmentation_factor=3):
        """
        数据增强
        
        Args:
            source_dir: 源数据目录
            target_dir: 目标数据目录
            augmentation_factor: 增强倍数
        """
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        
        images_dir = source_dir / 'images'
        labels_dir = source_dir / 'labels'
        
        target_images = target_dir / 'images'
        target_labels = target_dir / 'labels'
        
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue
            
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            shutil.copy(img_path, target_images / img_path.name)
            shutil.copy(label_path, target_labels / f"{img_path.stem}.txt")
            
            for i in range(augmentation_factor):
                augmented_img, aug_name = self.augment_single_image(img)
                aug_filename = f"{img_path.stem}_aug{i}{img_path.suffix}"
                
                cv2.imwrite(str(target_images / aug_filename), augmented_img)
                
                aug_label_path = target_labels / f"{img_path.stem}_aug{i}.txt"
                self._transform_label(label_path, aug_label_path, aug_name)
        
        print(f"数据增强完成！输出目录：{target_dir}")
    
    def augment_single_image(self, img):
        """
        单张图像增强
        
        Args:
            img: 输入图像
            
        Returns:
            tuple: (增强后的图像, 增强类型名称)
        """
        augmentations = [
            ('flip_h', lambda x: cv2.flip(x, 1)),
            ('rot_90_cw', lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)),
            ('rot_90_ccw', lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ('clahe', lambda x: self.enhance_image(x, 'clahe')),
        ]
        
        idx = np.random.randint(len(augmentations))
        aug_name, aug_func = augmentations[idx]
        return aug_func(img), aug_name
    
    def _transform_label(self, src_path, dst_path, aug_name):
        """
        根据增强类型变换标签坐标
        
        Args:
            src_path: 源标签文件路径
            dst_path: 目标标签文件路径
            aug_name: 增强类型名称
        """
        if aug_name == 'clahe':
            shutil.copy(src_path, dst_path)
            return
        
        with open(src_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            new_coords = []
            
            if aug_name == 'flip_h':
                for i in range(0, len(coords), 2):
                    new_coords.extend([1.0 - coords[i], coords[i + 1]])
            elif aug_name == 'rot_90_cw':
                for i in range(0, len(coords), 2):
                    new_coords.extend([coords[i + 1], 1.0 - coords[i]])
            elif aug_name == 'rot_90_ccw':
                for i in range(0, len(coords), 2):
                    new_coords.extend([1.0 - coords[i + 1], coords[i]])
            else:
                new_coords = coords
            
            new_lines.append(' '.join([str(class_id)] + [str(c) for c in new_coords]))
        
        with open(dst_path, 'w') as f:
            f.write('\n'.join(new_lines))
    
    def split_dataset(self, source_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        划分训练集、验证集和测试集
        
        Args:
            source_dir: 源数据目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        source_dir = Path(source_dir)
        images_dir = source_dir / 'images'
        labels_dir = source_dir / 'labels'
        
        image_files = list(images_dir.glob('*.jpg')) + \
                     list(images_dir.glob('*.png')) + \
                     list(images_dir.glob('*.jpeg'))
        
        train_files, temp_files = train_test_split(
            image_files,
            test_size=(1 - train_ratio),
            random_state=42
        )
        
        val_files, test_files = train_test_split(
            temp_files,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42
        )
        
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_images_dir = source_dir / split_name / 'images'
            split_labels_dir = source_dir / split_name / 'labels'
            
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in files:
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                shutil.copy(img_path, split_images_dir / img_path.name)
                
                if label_path.exists():
                    shutil.copy(label_path, split_labels_dir / f"{img_path.stem}.txt")
        
        print(f"数据集划分完成！")
        print(f"训练集：{len(train_files)} 张")
        print(f"验证集：{len(val_files)} 张")
        print(f"测试集：{len(test_files)} 张")
        
        return splits
    
    def convert_bbox_format(self, bbox, img_width, img_height, from_format='yolo', to_format='voc'):
        """
        边界框格式转换
        
        Args:
            bbox: 边界框坐标
            img_width: 图像宽度
            img_height: 图像高度
            from_format: 源格式 ('yolo' or 'voc')
            to_format: 目标格式 ('yolo' or 'voc')
            
        Returns:
            转换后的边界框
        """
        if from_format == to_format:
            return bbox
        
        if from_format == 'yolo':
            x_center, y_center, w, h = bbox
            x_center *= img_width
            y_center *= img_height
            w *= img_width
            h *= img_height
            
            x_min = x_center - w / 2
            y_min = y_center - h / 2
            x_max = x_center + w / 2
            y_max = y_center + h / 2
            
            if to_format == 'voc':
                return [x_min, y_min, x_max, y_max]
        
        elif from_format == 'voc':
            x_min, y_min, x_max, y_max = bbox
            w = x_max - x_min
            h = y_max - y_min
            x_center = x_min + w / 2
            y_center = y_min + h / 2
            
            if to_format == 'yolo':
                return [
                    x_center / img_width,
                    y_center / img_height,
                    w / img_width,
                    h / img_height
                ]
        
        return bbox
    
    def validate_dataset(self, dataset_dir):
        """
        验证数据集完整性
        
        Args:
            dataset_dir: 数据集目录
            
        Returns:
            验证报告
        """
        dataset_dir = Path(dataset_dir)
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        splits = {
            'train': (dataset_dir / 'images' / 'train', dataset_dir / 'labels' / 'train'),
            'val': (dataset_dir / 'images' / 'val', dataset_dir / 'labels' / 'val'),
            'test': (dataset_dir / 'test' / 'images', dataset_dir / 'test' / 'labels'),
        }
        
        for split, (images_dir, labels_dir) in splits.items():
            
            if not images_dir.exists():
                report['warnings'].append(f"缺少 images/{split} 目录")
                continue
            
            has_labels = labels_dir.exists()
            if not has_labels:
                report['warnings'].append(f"缺少 labels/{split} 目录")
            
            image_files = list(images_dir.glob('*.jpg')) + \
                         list(images_dir.glob('*.png'))
            label_files = list(labels_dir.glob('*.txt')) if has_labels else []
            
            report['stats'][split] = {
                'images': len(image_files),
                'labels': len(label_files)
            }
            
            if has_labels:
                for img_file in image_files:
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if not label_file.exists():
                        report['warnings'].append(f"缺少标签：{label_file}")
        
        return report


if __name__ == '__main__':
    datasets_dir = str(Path(__file__).parent.parent / 'datasets')
    preprocessor = CellDataPreprocessor(datasets_dir)
    
    report = preprocessor.validate_dataset(datasets_dir)
    print("数据集验证报告:")
    print(f"有效性：{report['valid']}")
    if report['errors']:
        print("错误:", report['errors'])
    if report['warnings']:
        print("警告:", report['warnings'])
    print("统计:", report['stats'])
