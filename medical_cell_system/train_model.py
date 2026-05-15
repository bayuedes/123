"""
医学细胞分类与计数系统 - 训练模块
基于 YOLOv8 实现细胞检测模型的训练
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


class CellModelTrainer:
    """细胞检测模型训练器"""
    
    def __init__(self, config=None):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典或 YAML 文件路径
        """
        if isinstance(config, str):
            with open(config, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config is None:
            self.config = self._get_default_config()
        else:
            self.config = config
        
        self.model = None
        self.results = None
        
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'model': 'yolov8m.pt',
            'data': None,
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'device': '0',
            'workers': 8,
            'optimizer': 'auto',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'project': 'runs/segment',
            'name': 'train',
            'exist_ok': False,
            'cache': 'disk',
            'seed': 42,
            'patience': 20,
            'save': True,
            'save_period': -1,
            'plots': True,
            'amp': True,
            'augment': True,
            'multi_scale': True,
        }
    
    def create_dataset_config(self, dataset_root, class_names, output_path='datasets/cell_data.yaml'):
        """
        创建数据集配置文件
        
        Args:
            dataset_root: 数据集根目录
            class_names: 类别名称列表
            output_path: 输出 YAML 文件路径
        """
        dataset_root = Path(dataset_root)
        
        config = {
            'path': str(dataset_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=None)
        
        print(f"数据集配置文件已创建：{output_path}")
        return output_path
    
    def train(self, override_config=None):
        """
        开始训练
        
        Args:
            override_config: 覆盖配置的字典
            
        Returns:
            训练结果
        """
        if override_config:
            self.config.update(override_config)
        
        print("=" * 60)
        print("开始训练医学细胞检测模型")
        print("=" * 60)
        print(f"模型：{self.config['model']}")
        print(f"数据集：{self.config['data']}")
        print(f" epochs: {self.config['epochs']}")
        print(f"图像尺寸：{self.config['imgsz']}")
        print(f"批次大小：{self.config['batch']}")
        print(f"设备：{self.config['device']}")
        print("=" * 60)
        
        model = YOLO(self.config['model'])
        
        train_args = {
            'data': self.config['data'],
            'epochs': self.config['epochs'],
            'imgsz': self.config['imgsz'],
            'batch': self.config['batch'],
            'device': self.config['device'],
            'workers': self.config['workers'],
            'optimizer': self.config['optimizer'],
            'lr0': self.config['lr0'],
            'lrf': self.config['lrf'],
            'momentum': self.config['momentum'],
            'weight_decay': self.config['weight_decay'],
            'warmup_epochs': self.config['warmup_epochs'],
            'warmup_momentum': self.config['warmup_momentum'],
            'box': self.config['box'],
            'cls': self.config['cls'],
            'dfl': self.config['dfl'],
            'project': self.config['project'],
            'name': self.config['name'],
            'exist_ok': self.config['exist_ok'],
            'cache': self.config['cache'],
            'seed': self.config['seed'],
            'patience': self.config['patience'],
            'save': self.config['save'],
            'save_period': self.config['save_period'],
            'plots': self.config['plots'],
            'amp': self.config['amp'],
            'augment': self.config['augment'],
            'multi_scale': self.config['multi_scale'],
        }
        
        train_args = {k: v for k, v in train_args.items() if v is not None}
        
        self.results = model.train(**train_args)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        
        return self.results
    
    def export_model(self, model_path, format='onnx', output_path=None):
        """
        导出模型
        
        Args:
            model_path: 模型权重文件路径
            format: 导出格式 ('onnx', 'torchscript', 'openvino', 'engine', etc.)
            output_path: 输出路径
            
        Returns:
            导出文件路径
        """
        print(f"正在导出模型到 {format} 格式...")
        
        model = YOLO(model_path)
        
        export_args = {
            'format': format,
        }
        
        if output_path:
            export_args['path'] = output_path
        
        exported_path = model.export(**export_args)
        
        print(f"模型已导出到：{exported_path}")
        return exported_path
    
    def validate_model(self, model_path, data=None):
        """
        验证模型性能
        
        Args:
            model_path: 模型权重文件路径
            data: 验证数据集配置文件路径
            
        Returns:
            验证结果
        """
        print("正在验证模型性能...")
        
        model = YOLO(model_path)
        
        val_args = {
            'data': data or self.config['data'],
            'split': 'val',
            'batch': self.config['batch'],
            'imgsz': self.config['imgsz'],
            'device': self.config['device'],
        }
        
        val_args = {k: v for k, v in val_args.items() if v is not None}
        
        metrics = model.val(**val_args)
        
        print("\n验证结果:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def get_training_stats(self):
        """获取训练统计信息"""
        if self.results is None:
            return None
        
        stats = {
            'epochs': self.results.epoch,
            'final_metrics': {
                'train/box_loss': float(self.results.results[-1][3]),
                'train/cls_loss': float(self.results.results[-1][4]),
                'train/dfl_loss': float(self.results.results[-1][5]),
                'metrics/precision_mAP': float(self.results.results[-1][6]),
                'metrics/recall_mAP': float(self.results.results[-1][7]),
                'metrics/mAP50': float(self.results.results[-1][8]),
                'metrics/mAP50-95': float(self.results.results[-1][9]),
            }
        }
        
        return stats


def train_cell_detection(
    dataset_config,
    model_name='yolov8m.pt',
    epochs=100,
    imgsz=640,
    batch=16,
    device='0',
    project='runs/segment',
    name='train',
    class_names=None
):
    """
    便捷函数：训练细胞检测模型
    
    Args:
        dataset_config: 数据集配置文件路径
        model_name: 预训练模型名称
        epochs: 训练轮数
        imgsz: 图像尺寸
        batch: 批次大小
        device: 设备
        project: 项目目录
        name: 实验名称
        class_names: 类别名称列表（用于自动创建配置）
        
    Returns:
        训练结果
    """
    trainer = CellModelTrainer()
    
    config = {
        'model': model_name,
        'data': dataset_config,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
    }
    
    return trainer.train(config)


if __name__ == '__main__':
    trainer = CellModelTrainer()
    
    dataset_config = trainer.create_dataset_config(
        dataset_root='datasets',
        class_names=['Platelets', 'RBC', 'WBC'],
        output_path='datasets/cell_data.yaml'
    )
    
    results = trainer.train({
        'model': 'yolov8m.pt',
        'data': str(dataset_config),
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': '0',
        'project': 'runs/segment',
        'name': 'experiment_1',
    })
    
    stats = trainer.get_training_stats()
    print("\n训练统计:")
    print(stats)
