"""
医学细胞分类与计数系统 - 系统测试脚本
验证系统各模块的基本功能
"""

import sys
from pathlib import Path


def test_imports():
    """测试导入"""
    print("测试 1: 检查模块导入...")
    
    try:
        from medical_cell_system import (
            CellDetector,
            CellModelTrainer,
            CellDataPreprocessor,
            MedicalCellSystem
        )
        print("  ✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"  ✗ 模块导入失败：{e}")
        return False


def test_preprocessing():
    """测试数据预处理"""
    print("\n测试 2: 检查数据预处理模块...")
    
    try:
        from medical_cell_system.data_preprocessing import CellDataPreprocessor
        
        preprocessor = CellDataPreprocessor('datasets')
        print("  ✓ 预处理器初始化成功")
        
        # 检查方法是否存在
        assert hasattr(preprocessor, 'enhance_image')
        assert hasattr(preprocessor, 'augment_dataset')
        assert hasattr(preprocessor, 'split_dataset')
        assert hasattr(preprocessor, 'validate_dataset')
        print("  ✓ 所有预处理方法可用")
        
        return True
    except Exception as e:
        print(f"  ✗ 预处理模块测试失败：{e}")
        return False


def test_inference():
    """测试推理模块"""
    print("\n测试 3: 检查推理模块...")
    
    try:
        from medical_cell_system.inference import CellDetector, CellCounter
        print("  ✓ 推理模块导入成功")
        
        # 检查类方法
        assert hasattr(CellDetector, 'detect')
        assert hasattr(CellDetector, 'detect_and_count')
        assert hasattr(CellDetector, 'batch_detect')
        assert hasattr(CellCounter, 'count_by_region')
        print("  ✓ 推理方法可用")
        
        return True
    except Exception as e:
        print(f"  ✗ 推理模块测试失败：{e}")
        return False


def test_training():
    """测试训练模块"""
    print("\n测试 4: 检查训练模块...")
    
    try:
        from medical_cell_system.train_model import CellModelTrainer
        
        trainer = CellModelTrainer()
        print("  ✓ 训练器初始化成功")
        
        assert hasattr(trainer, 'train')
        assert hasattr(trainer, 'validate_model')
        assert hasattr(trainer, 'export_model')
        print("  ✓ 训练方法可用")
        
        return True
    except Exception as e:
        print(f"  ✗ 训练模块测试失败：{e}")
        return False


def test_system():
    """测试系统整合模块"""
    print("\n测试 5: 检查系统整合模块...")
    
    try:
        from medical_cell_system.system import MedicalCellSystem
        
        system = MedicalCellSystem()
        print("  ✓ 系统初始化成功")
        
        assert hasattr(system, 'initialize')
        assert hasattr(system, 'detect_image')
        assert hasattr(system, 'export_statistics')
        assert hasattr(system, 'generate_report')
        print("  ✓ 系统方法可用")
        
        return True
    except Exception as e:
        print(f"  ✗ 系统整合模块测试失败：{e}")
        return False


def test_config():
    """测试配置文件"""
    print("\n测试 6: 检查配置文件...")
    
    config_path = Path(__file__).parent / 'config.yaml'
    
    if not config_path.exists():
        print(f"  ✗ 配置文件不存在：{config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert 'model' in config
        assert 'training' in config
        assert 'inference' in config
        print("  ✓ 配置文件格式正确")
        
        return True
    except Exception as e:
        print(f"  ✗ 配置文件测试失败：{e}")
        return False


def test_dependencies():
    """测试依赖包"""
    print("\n测试 7: 检查依赖包...")
    
    dependencies = [
        ('ultralytics', 'YOLOv8'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('yaml', 'PyYAML'),
    ]
    
    all_ok = True
    for package, name in dependencies:
        try:
            __import__(package)
            print(f"  ✓ {name} 已安装")
        except ImportError:
            print(f"  ✗ {name} 未安装")
            all_ok = False
    
    return all_ok


def test_gui_dependencies():
    """测试 GUI 依赖"""
    print("\n测试 8: 检查 GUI 依赖...")
    
    gui_deps = [
        ('PyQt5', 'PyQt5'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    all_ok = True
    for package, name in gui_deps:
        try:
            __import__(package)
            print(f"  ✓ {name} 已安装")
        except ImportError:
            print(f"  ✗ {name} 未安装 (可选)")
            # GUI 依赖是可选的
    
    return all_ok


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("医学细胞分类与计数系统 - 系统测试")
    print("=" * 60)
    
    tests = [
        ("依赖检查", test_dependencies),
        ("模块导入", test_imports),
        ("预处理模块", test_preprocessing),
        ("推理模块", test_inference),
        ("训练模块", test_training),
        ("系统整合", test_system),
        ("配置文件", test_config),
        ("GUI 依赖", test_gui_dependencies),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n测试 '{name}' 发生异常：{e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计：{passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统已就绪。")
    else:
        print("\n⚠️ 部分测试未通过，请检查相关模块。")
    
    print("=" * 60)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
