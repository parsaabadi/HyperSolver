#!/usr/bin/env python3
"""
Quick test script to verify HyperSolver installation and basic functionality
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    try:
        from src.model import ImprovedHyperGraphNet
        print("✅ HyperSolver model")
    except ImportError as e:
        print(f"❌ HyperSolver model: {e}")
        return False
    
    try:
        from src.trainer import train_model
        print("✅ HyperSolver trainer")
    except ImportError as e:
        print(f"❌ HyperSolver trainer: {e}")
        return False
    
    try:
        from src.data_reading import read_set_cover_instance
        print("✅ Data reading functions")
    except ImportError as e:
        print(f"❌ Data reading: {e}")
        return False
    
    return True

def test_data_files():
    """Test that sample data files exist"""
    print("\n📁 Testing data files...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("❌ Data directory not found")
        return False
    
    data_files = os.listdir(data_dir)
    if len(data_files) == 0:
        print("❌ No data files found")
        return False
    
    print(f"✅ Found {len(data_files)} data files")
    return True

def test_configs():
    """Test that configuration files exist"""
    print("\n⚙️ Testing configuration files...")
    
    config_dir = "configs"
    if not os.path.exists(config_dir):
        print("❌ Config directory not found")
        return False
    
    required_configs = [
        "set_cover_config.json",
        "subset_sum_config.json", 
        "hypermaxcut_config.json"
    ]
    
    for config in required_configs:
        if os.path.exists(os.path.join(config_dir, config)):
            print(f"✅ {config}")
        else:
            print(f"❌ {config} missing")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 HyperSolver Setup Test\n")
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test data files
    if not test_data_files():
        all_passed = False
    
    # Test configs
    if not test_configs():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! HyperSolver is ready to use.")
        print("\nTry running: python run.py --problem set_cover")
    else:
        print("❌ Some tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
