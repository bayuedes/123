"""
医学细胞分类与计数系统
基于 YOLOv8 的医学细胞智能检测与计数系统

功能特性:
- 支持多种血细胞类型检测 (RBC, WBC, Platelets 等)
- 提供完整的数据预处理和数据增强功能
- 支持模型训练、验证和导出
- 提供单图、批量、视频检测功能
- 现代化的图形用户界面
- 丰富的结果导出格式 (CSV, Excel, JSON, Markdown)
- 区域统计和高级分析功能

作者：Medical  Lab
版本：1.0.0
"""

from .inference import CellDetector, CellCounter, quick_detect
from .train_model import CellModelTrainer, train_cell_detection
from .data_preprocessing import CellDataPreprocessor
from .system import MedicalCellSystem

__version__ = '1.0.0'
__author__ = 'Medical AI Lab'

__all__ = [
    'CellDetector',
    'CellCounter',
    'CellModelTrainer',
    'CellDataPreprocessor',
    'MedicalCellSystem',
    'quick_detect',
    'train_cell_detection',
]
