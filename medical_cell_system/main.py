"""
医学细胞分类与计数系统 - 主程序入口
提供命令行接口和完整的工作流
"""

import sys
import argparse
import yaml
from pathlib import Path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='医学细胞分类与计数系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 单图检测
  python main.py detect --image images/test.jpg --model model.pt
  
  # 批量检测
  python main.py batch --images datasets/val/images --output results/
  
  # 训练模型
  python main.py train --data datasets/data.yaml --epochs 100
  
  # 启动 GUI
  python main.py gui
  
  # 导出结果
  python main.py export --results results.json --format csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 检测命令
    detect_parser = subparsers.add_parser('detect', help='检测单张图片')
    detect_parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    detect_parser.add_argument('--model', type=str, default='runs/segment/train/weights/best.pt', help='模型路径')
    detect_parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    detect_parser.add_argument('--output', type=str, default='results/', help='输出目录')
    detect_parser.add_argument('--save', action='store_true', help='保存结果')
    detect_parser.add_argument('--show', action='store_true', help='显示结果')
    
    # 批量检测命令
    batch_parser = subparsers.add_parser('batch', help='批量检测')
    batch_parser.add_argument('--images', type=str, required=True, help='图片目录')
    batch_parser.add_argument('--model', type=str, default='runs/segment/train/weights/best.pt', help='模型路径')
    batch_parser.add_argument('--output', type=str, default='results/batch/', help='输出目录')
    batch_parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    batch_parser.add_argument('--export', type=str, default='csv', choices=['csv', 'json', 'excel'], help='导出格式')
    
    # 视频检测命令
    video_parser = subparsers.add_parser('video', help='视频检测')
    video_parser.add_argument('--video', type=str, required=True, help='视频路径')
    video_parser.add_argument('--model', type=str, default='runs/segment/train/weights/best.pt', help='模型路径')
    video_parser.add_argument('--output', type=str, help='输出视频路径')
    video_parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data', type=str, required=True, help='数据集配置文件')
    train_parser.add_argument('--model', type=str, default='yolov8m.pt', help='预训练模型')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    train_parser.add_argument('--batch', type=int, default=16, help='批次大小')
    train_parser.add_argument('--device', type=str, default='0', help='GPU 设备')
    train_parser.add_argument('--project', type=str, default='runs/segment', help='项目目录')
    train_parser.add_argument('--name', type=str, default='train', help='实验名称')
    
    # 验证命令
    val_parser = subparsers.add_parser('val', help='验证模型')
    val_parser.add_argument('--model', type=str, required=True, help='模型路径')
    val_parser.add_argument('--data', type=str, required=True, help='数据集配置文件')
    val_parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    val_parser.add_argument('--batch', type=int, default=16, help='批次大小')
    val_parser.add_argument('--device', type=str, default='0', help='GPU 设备')
    
    # 导出命令
    export_parser = subparsers.add_parser('export', help='导出模型')
    export_parser.add_argument('--model', type=str, required=True, help='模型路径')
    export_parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'torchscript', 'openvino'], help='导出格式')
    export_parser.add_argument('--output', type=str, help='输出路径')
    
    # GUI 命令
    gui_parser = subparsers.add_parser('gui', help='启动图形界面')
    gui_parser.add_argument('--model', type=str, help='模型路径')
    
    # 信息命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    
    # 解析参数
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # 执行命令
    if args.command == 'detect':
        cmd_detect(args)
    elif args.command == 'batch':
        cmd_batch(args)
    elif args.command == 'video':
        cmd_video(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'val':
        cmd_val(args)
    elif args.command == 'export':
        cmd_export(args)
    elif args.command == 'gui':
        cmd_gui(args)
    elif args.command == 'info':
        cmd_info(args)


def cmd_detect(args):
    """检测单张图片"""
    from medical_cell_system.inference import CellDetector
    
    print(f"加载模型：{args.model}")
    detector = CellDetector(
        model_path=args.model,
        class_names=['Platelets', 'RBC', 'WBC'],
        conf_threshold=args.conf
    )
    
    print(f"检测图片：{args.image}")
    result = detector.detect_and_count(args.image, conf_threshold=args.conf)
    
    print("\n检测结果:")
    for cls_id, count in result['count_stats'].items():
        print(f"  {detector.class_names[cls_id]}: {count}")
    print(f"总计：{result['total_count']}")
    
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"result_{Path(args.image).name}"
        detector.save_results(args.image, output_path, detector.detect(args.image))
        print(f"结果已保存到：{output_path}")


def cmd_batch(args):
    """批量检测"""
    from medical_cell_system.inference import CellDetector
    
    print(f"加载模型：{args.model}")
    detector = CellDetector(
        model_path=args.model,
        class_names=['Platelets', 'RBC', 'WBC'],
        conf_threshold=args.conf
    )
    
    print(f"批量检测目录：{args.images}")
    results = detector.batch_detect(
        image_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf
    )
    
    print(f"\n批量检测完成!")
    print(f"处理图像数：{results['image_count']}")
    print(f"总细胞数：{results['total_count']}")
    
    for cls_id, count in results['total_stats'].items():
        print(f"  {detector.class_names[cls_id]}: {count}")


def cmd_video(args):
    """视频检测"""
    from medical_cell_system.inference import CellDetector
    
    print(f"加载模型：{args.model}")
    detector = CellDetector(
        model_path=args.model,
        class_names=['Platelets', 'RBC', 'WBC'],
        conf_threshold=args.conf
    )
    
    print(f"检测视频：{args.video}")
    results = detector.detect_video(
        video_path=args.video,
        output_path=args.output,
        conf_threshold=args.conf
    )
    
    print(f"\n视频检测完成!")
    print(f"处理帧数：{results['frame_count']}")
    if results['video_path']:
        print(f"输出视频：{results['video_path']}")


def cmd_train(args):
    """训练模型"""
    from medical_cell_system.train_model import CellModelTrainer
    
    trainer = CellModelTrainer()
    
    print(f"使用数据集：{args.data}")
    print(f"预训练模型：{args.model}")
    print(f"训练轮数：{args.epochs}")
    print(f"图像尺寸：{args.imgsz}")
    print(f"批次大小：{args.batch}")
    print(f"设备：{args.device}")
    
    results = trainer.train({
        'model': args.model,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'project': args.project,
        'name': args.name,
    })
    
    print(f"\n训练完成!")
    print(f"模型保存在：{trainer.config['project']}/{trainer.config['name']}/weights/")


def cmd_val(args):
    """验证模型"""
    from medical_cell_system.train_model import CellModelTrainer
    
    trainer = CellModelTrainer()
    
    print(f"验证模型：{args.model}")
    metrics = trainer.validate_model(
        model_path=args.model,
        data=args.data
    )
    
    print(f"\n验证结果:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


def cmd_export(args):
    """导出模型"""
    from medical_cell_system.train_model import CellModelTrainer
    
    trainer = CellModelTrainer()
    
    print(f"导出模型：{args.model}")
    print(f"导出格式：{args.format}")
    
    exported_path = trainer.export_model(
        model_path=args.model,
        format=args.format,
        output_path=args.output
    )
    
    print(f"模型已导出到：{exported_path}")


def cmd_gui(args):
    """启动 GUI"""
    from medical_cell_system.gui import main
    
    print("启动图形用户界面...")
    main()


def cmd_info(args):
    """显示系统信息"""
    import torch
    from ultralytics import __version__ as ultralytics_version
    
    print("=" * 60)
    print("医学细胞分类与计数系统 - 系统信息")
    print("=" * 60)
    print(f"Ultralytics 版本：{ultralytics_version}")
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本：{torch.version.cuda}")
        print(f"GPU 数量：{torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
