"""
医学细胞分类与计数系统 - 主程序整合模块
提供统一的系统接口和完整的 workflows
"""

import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


class MedicalCellSystem:
    """医学细胞分类与计数系统主类"""
    
    def __init__(self, config_path=None):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.initialized = False
        self.detector = None
        self.trainer = None
        
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'system': {
                'name': '医学细胞分类与计数系统',
                'version': '1.0.0',
                'author': 'Medical AI Lab'
            },
            'model': {
                'path': 'runs/segment/train/weights/best.pt',
                'class_names': ['Platelets', 'RBC', 'WBC'],
                'conf_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'training': {
                'epochs': 100,
                'imgsz': 640,
                'batch': 16,
                'device': '0'
            },
            'inference': {
                'save_results': True,
                'show_visualization': True,
                'output_dir': 'results'
            }
        }
    
    def initialize(self, model_path=None):
        """
        初始化系统
        
        Args:
            model_path: 模型路径（可选）
        """
        from .inference import CellDetector
        from .train_model import CellModelTrainer
        
        model_path = model_path or self.config['model']['path']
        
        if not Path(model_path).exists():
            print(f"警告：模型文件不存在：{model_path}")
            print("请先训练模型或指定正确的模型路径")
            return False
        
        self.detector = CellDetector(
            model_path=model_path,
            class_names=self.config['model']['class_names'],
            conf_threshold=self.config['model']['conf_threshold'],
            iou_threshold=self.config['model']['iou_threshold']
        )
        
        self.trainer = CellModelTrainer()
        self.initialized = True
        
        print("✓ 系统初始化成功")
        print(f"  模型：{model_path}")
        print(f"  类别：{self.config['model']['class_names']}")
        print(f"  置信度阈值：{self.config['model']['conf_threshold']}")
        
        return True
    
    def train(self, dataset_config, **kwargs):
        """
        训练模型
        
        Args:
            dataset_config: 数据集配置文件路径
            **kwargs: 其他训练参数
        """
        if not self.trainer:
            from .train_model import CellModelTrainer
            self.trainer = CellModelTrainer()
        
        config = self.config['training'].copy()
        config['data'] = dataset_config
        config.update(kwargs)
        
        print("=" * 60)
        print("开始训练模型")
        print("=" * 60)
        
        results = self.trainer.train(config)
        
        self.config['model']['path'] = self.trainer.config['project'] + '/' + \
                                       self.trainer.config['name'] + '/weights/best.pt'
        
        print("\n✓ 训练完成")
        print(f"模型保存路径：{self.config['model']['path']}")
        
        return results
    
    def detect_image(self, image_path, save_result=False, show_result=True):
        """
        检测单张图像
        
        Args:
            image_path: 图像路径
            save_result: 是否保存结果
            show_result: 是否显示结果
            
        Returns:
            检测结果
        """
        if not self.initialized:
            raise RuntimeError("系统未初始化，请先调用 initialize()")
        
        result = self.detector.detect_and_count(
            image_path,
            conf_threshold=self.config['model']['conf_threshold']
        )
        
        if save_result:
            output_dir = Path(self.config['inference']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"result_{Path(image_path).name}"
            self.detector.save_results(
                image_path,
                output_path,
                self.detector.detect(image_path)
            )
        
        if show_result:
            self._print_result(result)
        
        return result
    
    def detect_batch(self, image_dir, output_dir=None):
        """
        批量检测
        
        Args:
            image_dir: 图像目录
            output_dir: 输出目录
            
        Returns:
            批量检测结果
        """
        if not self.initialized:
            raise RuntimeError("系统未初始化")
        
        output_dir = output_dir or self.config['inference']['output_dir']
        
        results = self.detector.batch_detect(
            image_dir,
            output_dir,
            conf_threshold=self.config['model']['conf_threshold']
        )
        
        print(f"\n批量检测完成")
        print(f"处理图像数：{results['image_count']}")
        print(f"总细胞数：{results['total_count']}")
        
        return results
    
    def detect_video(self, video_path, output_path=None):
        """
        检测视频
        
        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            
        Returns:
            视频检测结果
        """
        if not self.initialized:
            raise RuntimeError("系统未初始化")
        
        results = self.detector.detect_video(
            video_path,
            output_path,
            conf_threshold=self.config['model']['conf_threshold']
        )
        
        return results
    
    def export_statistics(self, results, output_path='statistics.csv', format='csv'):
        """
        导出统计结果
        
        Args:
            results: 检测结果
            output_path: 输出文件路径
            format: 输出格式 ('csv', 'json', 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            if 'per_image_stats' in results:
                data = []
                for stat in results['per_image_stats']:
                    row = {'filename': stat['filename']}
                    for cls_id, count in stat['stats']['count_stats'].items():
                        row[f'class_{cls_id}'] = count
                    row['total'] = stat['stats']['total_count']
                    data.append(row)
                
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            else:
                data = []
                for cls_id, count in results['count_stats'].items():
                    data.append({
                        'class_id': cls_id,
                        'class_name': self.detector.class_names.get(cls_id, f'Class {cls_id}'),
                        'count': count
                    })
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        elif format == 'excel':
            if 'per_image_stats' in results:
                data = []
                for stat in results['per_image_stats']:
                    row = {'filename': stat['filename']}
                    for cls_id, count in stat['stats']['count_stats'].items():
                        row[f'class_{cls_id}'] = count
                    row['total'] = stat['stats']['total_count']
                    data.append(row)
                
                df = pd.DataFrame(data)
                df.to_excel(output_path, index=False, engine='openpyxl')
        
        print(f"统计结果已导出到：{output_path}")
        return output_path
    
    def _print_result(self, result):
        """打印检测结果"""
        print("\n" + "=" * 60)
        print("检测结果")
        print("=" * 60)
        
        if 'count_stats' in result:
            print("\n细胞分类统计:")
            for cls_id, count in result['count_stats'].items():
                class_name = self.detector.class_names.get(cls_id, f'Class {cls_id}')
                print(f"  {class_name}: {count}")
            
            print(f"\n总计：{result['total_count']} 个细胞")
        
        if 'image_shape' in result:
            h, w = result['image_shape'][:2]
            print(f"\n图像尺寸：{w} x {h}")
        
        print("=" * 60)
    
    def generate_report(self, results, output_path='report.md'):
        """
        生成 Markdown 格式报告
        
        Args:
            results: 检测结果
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# 医学细胞检测报告\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**模型**: {self.config['model']['path']}\n")
        report.append(f"**置信度阈值**: {self.config['model']['conf_threshold']}\n\n")
        
        report.append("## 检测结果\n\n")
        
        if 'count_stats' in results:
            report.append("### 细胞统计\n\n")
            report.append("| 类别 | 数量 |\n")
            report.append("|------|------|\n")
            for cls_id, count in results['count_stats'].items():
                class_name = self.detector.class_names.get(cls_id, f'Class {cls_id}')
                report.append(f"| {class_name} | {count} |\n")
            
            report.append(f"\n**总计**: {results['total_count']} 个细胞\n")
        
        if 'per_image_stats' in results:
            report.append("\n### 各图像统计\n\n")
            report.append("| 文件名 | ")
            for cls_name in self.detector.class_names.values():
                report.append(f"{cls_name} | ")
            report.append("总计 |\n")
            
            report.append("|" + "|".join(["------"] * (len(self.detector.class_names) + 2)) + "|\n")
            
            for stat in results['per_image_stats']:
                report.append(f"| {stat['filename']} |")
                for cls_id in sorted(stat['stats']['count_stats'].keys()):
                    count = stat['stats']['count_stats'].get(cls_id, 0)
                    report.append(f" {count} |")
                report.append(f" {stat['stats']['total_count']} |\n")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("".join(report))
        
        print(f"报告已保存到：{output_path}")
        return output_path
    
    def save_config(self, output_path='config.yaml'):
        """保存配置"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=None)
        
        print(f"配置已保存到：{output_path}")
    
    def get_system_info(self):
        """获取系统信息"""
        import torch
        from ultralytics import __version__
        
        info = {
            'system_name': self.config['system']['name'],
            'version': self.config['system']['version'],
            'ultralytics_version': __version__,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'model_path': self.config['model']['path'],
            'class_names': self.config['model']['class_names']
        }
        
        return info


def main():
    """主函数示例"""
    print("=" * 60)
    print("医学细胞分类与计数系统")
    print("=" * 60)
    
    system = MedicalCellSystem()
    
    if not system.initialize():
        print("\n系统初始化失败，请检查模型文件")
        return
    
    system_info = system.get_system_info()
    print("\n系统信息:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    print("\n系统已就绪，可以进行检测或训练")


if __name__ == '__main__':
    main()
