"""
医学细胞分类与计数系统 - Gradio Web 界面
基于 Gradio 的现代化 Web 界面，提供细胞检测、计数和统计分析功能
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import cv2
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from gradio.cli.commands.sketch import launch
from ultralytics import YOLO

import gradio as gr


class MedicalCellGradioApp:
    """医学细胞检测 Gradio 应用"""
    
    def __init__(self, config_path=None):
        """
        初始化应用
        
        Args:
            config_path: 配置文件路径
        """
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config(config_path)
        self.model = None
        self.model_path = None
        self.class_names = self.config.get('model', {}).get('class_names', ['Platelets', 'RBC', 'WBC'])
        self.last_results = None
        
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        default_config = {
            'model': {
                'path': r'D:\code\demo\ultralytics-8.3.163\runs\segment\train\weights\best.pt',
                'class_names': ['Platelets', 'RBC', 'WBC'],
                'conf_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'inference': {
                'output_dir': 'results'
            }
        }
        return default_config
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            str: 状态信息
        """
        try:
            path = Path(model_path)
            
            if not path.is_absolute():
                cwd_path = Path.cwd() / path
                project_path = self.project_root / path
                
                if cwd_path.exists():
                    path = cwd_path
                elif project_path.exists():
                    path = project_path
                else:
                    return f"❌ 模型文件不存在：{model_path}\n尝试过的路径：\n  - {cwd_path}\n  - {project_path}"
            
            if not path.exists():
                return f"❌ 模型文件不存在：{path}"
            
            self.model = YOLO(str(path))
            self.model_path = str(path)
            return f"✅ 模型加载成功：{path.name}"
        except Exception as e:
            return f"❌ 模型加载失败：{str(e)}"
    
    def detect_image(self, image, conf_threshold):
        """
        检测单张图片
        
        Args:
            image: 输入图片（numpy 数组或 None）
            conf_threshold: 置信度阈值
            
        Returns:
            tuple: (处理后的图片，检测结果文本，统计图表)
        """
        if image is None:
            return None, "❌ 请先上传图片", None
        
        if self.model is None:
            return None, "❌ 请先加载模型", None
        
        try:
            results = self.model.predict(image, conf=conf_threshold, verbose=False)
            
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
                        'class': cls
                    })
            
            self.last_results = {
                'detections': detections,
                'count_stats': dict(count_stats),
                'total_count': sum(count_stats.values())
            }
            
            img = image.copy() if isinstance(image, np.ndarray) else cv2.imread(image)
            
            color_map = {
                0: (0, 255, 0),
                1: (0, 0, 255),
                2: (255, 0, 0)
            }
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                conf = det['confidence']
                cls = det['class']
                
                color = color_map.get(cls, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label = f"{self.class_names[cls] if cls < len(self.class_names) else f'Class {cls}'} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            result_text = self._format_result_text(self.last_results)
            chart = self._create_statistics_chart(self.last_results)
            
            return img, result_text, chart
            
        except Exception as e:
            return None, f"❌ 检测失败：{str(e)}", None
    
    def _format_result_text(self, results):
        """格式化检测结果文本"""
        lines = ["=" * 50, "🔬 医学细胞检测结果", "=" * 50, ""]
        
        if 'count_stats' in results:
            lines.append("📊 细胞分类统计:")
            for cls_id, count in results['count_stats'].items():
                class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
                lines.append(f"  • {class_name}: {count}")
            
            lines.append("")
            lines.append(f"💯 总计：{results['total_count']} 个细胞")
        
        lines.append("")
        lines.append(f"⏰ 检测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def _create_statistics_chart(self, results):
        """创建统计图表"""
        if 'count_stats' not in results:
            return None
        
        labels = []
        values = []
        for cls_id in sorted(results['count_stats'].keys()):
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
            labels.append(class_name)
            values.append(results['count_stats'][cls_id])
        
        if not values:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        bars = ax1.bar(labels, values, color=colors[:len(labels)])
        ax1.set_title('细胞数量统计', fontsize=14, fontweight='bold')
        ax1.set_xlabel('细胞类型')
        ax1.set_ylabel('数量')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value}', ha='center', va='bottom', fontsize=10)
        
        if sum(values) > 0:
            wedges, texts, autotexts = ax2.pie(values, labels=labels, autopct='%1.1f%%',
                                               colors=colors[:len(labels)], startangle=90)
            ax2.set_title('细胞分布比例', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def batch_detect(self, folder, conf_threshold, progress=gr.Progress()):
        """
        批量检测
        
        Args:
            folder: 包含图片的文件夹
            conf_threshold: 置信度阈值
            progress: Gradio 进度对象
            
        Returns:
            tuple: (统计表格，统计图表，结果文本)
        """
        if folder is None:
            return None, None, "❌ 请选择文件夹"
        
        if self.model is None:
            return None, None, "❌ 请先加载模型"
        
        try:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = [f for f in Path(folder).iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            if not image_files:
                return None, None, f"❌ 目录中没有找到图片：{folder}"
            
            progress(0, desc="开始批量检测...")
            
            results_data = []
            total_stats = defaultdict(int)
            
            for idx, img_path in enumerate(image_files):
                progress((idx + 1) / len(image_files), desc=f"处理中：{img_path.name}")
                
                result = self.model.predict(str(img_path), conf=conf_threshold, verbose=False)
                
                row_stats = defaultdict(int)
                for res in result:
                    if res.boxes:
                        for cls in res.boxes.cls.cpu().numpy():
                            cls_id = int(cls)
                            row_stats[cls_id] += 1
                            total_stats[cls_id] += 1
                
                row = {'文件名': img_path.name}
                for i in range(len(self.class_names)):
                    row[self.class_names[i]] = row_stats.get(i, 0)
                row['总计'] = sum(row_stats.values())
                results_data.append(row)
            
            df = pd.DataFrame(results_data)
            
            total_row = {'文件名': '总计'}
            for i in range(len(self.class_names)):
                total_row[self.class_names[i]] = total_stats.get(i, 0)
            total_row['总计'] = sum(total_stats.values())
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
            
            chart = self._create_batch_chart(total_stats)
            
            result_text = self._format_batch_result(total_stats, len(image_files))
            
            return df, chart, result_text
            
        except Exception as e:
            return None, None, f"❌ 批量检测失败：{str(e)}"
    
    def _create_batch_chart(self, total_stats):
        """创建批量检测统计图表"""
        labels = []
        values = []
        for cls_id in sorted(total_stats.keys()):
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
            labels.append(class_name)
            values.append(total_stats[cls_id])
        
        if not values:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        bars = ax1.bar(labels, values, color=colors[:len(labels)])
        ax1.set_title('批量检测 - 细胞数量统计', fontsize=14, fontweight='bold')
        ax1.set_xlabel('细胞类型')
        ax1.set_ylabel('数量')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value}', ha='center', va='bottom', fontsize=10)
        
        if sum(values) > 0:
            ax2.pie(values, labels=labels, autopct='%1.1f%%',
                   colors=colors[:len(labels)], startangle=90)
            ax2.set_title('细胞分布比例', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _format_batch_result(self, total_stats, image_count):
        """格式化批量检测结果"""
        lines = ["=" * 50, "📁 批量检测结果", "=" * 50, ""]
        lines.append(f"📊 处理图片数：{image_count}")
        lines.append("")
        lines.append("🔬 细胞分类统计:")
        
        for cls_id, count in total_stats.items():
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
            lines.append(f"  • {class_name}: {count}")
        
        lines.append("")
        lines.append(f"💯 总细胞数：{sum(total_stats.values())}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def export_results(self, results_data, format_type):
        """
        导出检测结果
        
        Args:
            results_data: 结果数据
            format_type: 导出格式
            
        Returns:
            str: 导出文件路径
        """
        if results_data is None:
            return "❌ 没有可导出的数据"
        
        try:
            output_dir = Path(self.config.get('inference', {}).get('output_dir', 'results'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format_type == 'CSV':
                output_path = output_dir / f'cell_detection_{timestamp}.csv'
                results_data.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            elif format_type == 'Excel':
                output_path = output_dir / f'cell_detection_{timestamp}.xlsx'
                results_data.to_excel(output_path, index=False, engine='openpyxl')
            
            elif format_type == 'JSON':
                output_path = output_dir / f'cell_detection_{timestamp}.json'
                export_data = results_data.to_dict(orient='records')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            else:
                return "❌ 不支持的导出格式"
            
            return f"✅ 结果已导出到：{output_path}"
            
        except Exception as e:
            return f"❌ 导出失败：{str(e)}"
    
    def create_gradio_interface(self):
        """创建 Gradio 界面"""
        with gr.Blocks(title="医学细胞分类与计数系统", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🔬 医学细胞分类与计数系统
                
                基于 YOLOv8 的医学细胞智能检测与计数平台
                
                ---
                """
            )
            
            with gr.Tabs():
                with gr.TabItem("🖼️ 单图检测"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📤 上传图片")
                            image_input = gr.Image(
                                label="上传医学细胞图片",
                                type="numpy"
                            )
                            
                            gr.Markdown("### ⚙️ 检测配置")
                            conf_slider = gr.Slider(
                                minimum=0.01,
                                maximum=1.0,
                                value=0.25,
                                step=0.01,
                                label="置信度阈值"
                            )
                            
                            detect_btn = gr.Button(
                                "🚀 开始检测",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 📊 检测结果")
                            image_output = gr.Image(
                                label="检测结果",
                                type="numpy"
                            )
                            
                    with gr.Row():
                        with gr.Column(scale=1):
                            result_text = gr.Textbox(
                                label="检测统计",
                                lines=10,
                                max_lines=20
                            )
                        
                        with gr.Column(scale=1):
                            chart_output = gr.Plot(
                                label="统计图表"
                            )
                
                with gr.TabItem("📁 批量检测"):
                    gr.Markdown("### 📂 选择图片文件夹")
                    
                    with gr.Row():
                        folder_input = gr.Textbox(
                            label="图片文件夹路径",
                            placeholder="输入包含医学细胞图片的文件夹路径"
                        )
                        
                        folder_btn = gr.Button(
                            "📂 浏览文件夹",
                            variant="secondary"
                        )
                    
                    with gr.Row():
                        batch_conf_slider = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=0.25,
                            step=0.01,
                            label="置信度阈值"
                        )
                        
                        batch_detect_btn = gr.Button(
                            "▶️ 开始批量检测",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            batch_table = gr.Dataframe(
                                label="检测结果统计"
                            )
                        
                        with gr.Column(scale=1):
                            batch_chart = gr.Plot(
                                label="批量检测统计"
                            )
                    
                    batch_result_text = gr.Textbox(
                        label="检测摘要",
                        lines=10,
                        max_lines=15
                    )
                    
                    with gr.Row():
                        export_format = gr.Dropdown(
                            choices=["CSV", "Excel", "JSON"],
                            value="CSV",
                            label="导出格式"
                        )
                        
                        export_btn = gr.Button(
                            "💾 导出结果",
                            variant="primary"
                        )
                    
                    export_status = gr.Textbox(
                        label="导出状态",
                        interactive=False
                    )
                
                with gr.TabItem("ℹ️ 系统信息"):
                    gr.Markdown("### 📋 系统信息")
                    
                    info_text = gr.Textbox(
                        value=self._get_system_info(),
                        label="系统信息",
                        lines=15,
                        interactive=False
                    )
                    
                    gr.Markdown("""
                    ### 📖 使用说明
                    
                    1. **单图检测**:
                       - 上传医学细胞图片
                       - 调整置信度阈值（可选）
                       - 点击"开始检测"按钮
                       - 查看检测结果和统计图表
                    
                    2. **批量检测**:
                       - 输入包含图片的文件夹路径
                       - 调整置信度阈值（可选）
                       - 点击"开始批量检测"按钮
                       - 查看统计表格和图表
                       - 导出检测结果
                    
                    3. **结果导出**:
                       - 支持 CSV、Excel、JSON 格式
                       - 结果自动保存到 results 目录
                    """)
            
            model_path_input = gr.Textbox(
                label="模型路径",
                value=self.config.get('model', {}).get('path', ''),
                visible=True
            )
            
            load_model_btn = gr.Button(
                "📥 加载模型",
                variant="secondary"
            )
            
            model_status = gr.Textbox(
                label="模型状态",
                interactive=False
            )
            
            load_model_btn.click(
                fn=self.load_model,
                inputs=[model_path_input],
                outputs=[model_status]
            )
            
            detect_btn.click(
                fn=self.detect_image,
                inputs=[image_input, conf_slider],
                outputs=[image_output, result_text, chart_output]
            )
            
            batch_detect_btn.click(
                fn=self.batch_detect,
                inputs=[folder_input, batch_conf_slider],
                outputs=[batch_table, batch_chart, batch_result_text]
            )
            
            export_btn.click(
                fn=self.export_results,
                inputs=[batch_table, export_format],
                outputs=[export_status]
            )
            
            folder_btn.click(
                fn=lambda: gr.Info("请在文件资源管理器中选择文件夹，然后复制路径粘贴到输入框"),
                inputs=[],
                outputs=[]
            )
        
        return demo
    
    def _get_system_info(self):
        """获取系统信息"""
        import torch
        from ultralytics import __version__ as ultralytics_version
        
        info_lines = [
            "=" * 50,
            "医学细胞分类与计数系统",
            "=" * 50,
            "",
            f"📦 Ultralytics 版本：{ultralytics_version}",
            f"🔥 PyTorch 版本：{torch.__version__}",
            f"💻 CUDA 可用：{torch.cuda.is_available()}",
        ]
        
        if torch.cuda.is_available():
            info_lines.extend([
                f"  • CUDA 版本：{torch.version.cuda}",
                f"  • GPU 数量：{torch.cuda.device_count()}",
            ])
            for i in range(torch.cuda.device_count()):
                info_lines.append(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
        
        info_lines.extend([
            "",
            f"🎯 类别数量：{len(self.class_names)}",
            f"📝 类别名称：{', '.join(self.class_names)}",
            "",
            "=" * 50,
        ])
        
        return "\n".join(info_lines)


def main():
    """主函数"""
    print("=" * 60)
    print("医学细胞分类与计数系统 - Gradio Web 界面")
    print("=" * 60)
    
    config_path = Path(__file__).parent / 'config.yaml'
    app = MedicalCellGradioApp(config_path=str(config_path))
    
    demo = app.create_gradio_interface()
    
    print("\n启动 Web 服务...")
    print("请在浏览器中访问显示的地址")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == '__main__':
    main()
