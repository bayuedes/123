"""
PyTorch DLL 加载问题修复脚本
"""

import os
import sys
import subprocess
from pathlib import Path


def fix_dll_blocking():
    """方法 1: 解除 DLL 文件阻止"""
    print("=" * 60)
    print("方法 1: 解除 DLL 文件阻止")
    print("=" * 60)
    
    # 获取 torch 路径
    try:
        import torch
        torch_path = Path(torch.__file__).parent
        torch_lib_path = torch_path / 'lib'
        
        print(f"PyTorch 路径：{torch_path}")
        print(f"PyTorch lib 路径：{torch_lib_path}")
        
        # 使用 PowerShell 取消阻止所有 DLL 文件
        cmd = f'Get-ChildItem -Path "{torch_lib_path}\\*.dll" -Recurse | Unblock-File'
        print(f"执行：{cmd}")
        
        result = subprocess.run(
            ['powershell', '-Command', cmd],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ DLL 文件已取消阻止")
            return True
        else:
            print(f"✗ 取消阻止失败：{result.stderr}")
            return False
    except ImportError:
        print("✗ 无法导入 PyTorch")
        return False


def set_environment_variables():
    """方法 2: 设置环境变量"""
    print("\n" + "=" * 60)
    print("方法 2: 设置环境变量")
    print("=" * 60)
    
    env_vars = {
        'TORCH_DLL_LOAD_TIMEOUT': '30',
        'PYTORCH_ENABLE_FALLBACK_LOADING': '1',
        'KMP_DUPLICATE_LIB_OK': 'True',
    }
    
    print("设置环境变量...")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  ✓ {key} = {value}")
    
    print("✓ 环境变量已设置")
    return True


def reinstall_pytorch_cpu():
    """方法 3: 重新安装 CPU 版本 PyTorch"""
    print("\n" + "=" * 60)
    print("方法 3: 重新安装 PyTorch (CPU 版本)")
    print("=" * 60)
    print("如果其他方法无效，可以执行以下命令重新安装:")
    print()
    print("pip uninstall torch torchvision torchaudio -y")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print()
    
    choice = input("是否现在重新安装 PyTorch? (y/n): ")
    if choice.lower() == 'y':
        print("\n卸载 PyTorch...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'])
        
        print("\n重新安装 PyTorch (CPU 版本)...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ])
        
        print("\n✓ PyTorch 重新安装完成")
        return True
    
    return False


def check_windows_defender():
    """检查 Windows Defender 设置"""
    print("\n" + "=" * 60)
    print("方法 4: 检查 Windows Defender 设置")
    print("=" * 60)
    print()
    print("Windows Defender 的应用程序控制策略可能阻止了 DLL 加载。")
    print("解决方法:")
    print("  1. 打开 Windows 安全中心")
    print("  2. 点击'应用和浏览器控制'")
    print("  3. 点击'基于声誉的保护设置'")
    print("  4. 关闭'可能不需要的应用阻止'（临时）")
    print("  5. 重新运行程序")
    print()
    print("或者将以下路径添加到排除列表:")
    
    try:
        import torch
        torch_path = Path(torch.__file__).parent
        print(f"  - {torch_path}")
    except:
        print("  - 你的 conda 环境路径")


def test_import():
    """测试导入"""
    print("\n" + "=" * 60)
    print("测试 PyTorch 导入")
    print("=" * 60)
    
    try:
        print("正在导入 torch...")
        import torch
        print(f"✓ torch 导入成功 (版本 {torch.__version__})")
        
        print("正在导入 ultralytics...")
        from ultralytics import YOLO
        print("✓ ultralytics 导入成功")
        
        print("\n所有导入成功！")
        return True
    except Exception as e:
        print(f"✗ 导入失败：{e}")
        return False


def main():
    print("=" * 60)
    print("PyTorch DLL 加载问题修复工具")
    print("=" * 60)
    print()
    
    # 先测试当前状态
    print("测试当前状态...")
    if test_import():
        print("\n✓ PyTorch 已正常工作，无需修复")
        return
    
    print("\nPyTorch 加载失败，开始修复...")
    print()
    
    # 执行修复步骤
    fix_dll_blocking()
    set_environment_variables()
    
    # 再次测试
    print("\n" + "=" * 60)
    print("修复后测试")
    print("=" * 60)
    
    if test_import():
        print("\n✓ 修复成功！PyTorch 现在可以正常工作了")
        print("\n现在可以运行 gradio_app.py")
        return
    
    print("\n✗ 修复未成功")
    print("\n建议的下一步操作:")
    print("  1. 检查 Windows Defender 设置")
    print("  2. 重新安装 PyTorch")
    print("  3. 使用管理员权限运行命令提示符")
    print()
    
    check_windows_defender()


if __name__ == '__main__':
    main()
