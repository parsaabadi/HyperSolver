#!/usr/bin/env python3
"""
Quick test script to verify HyperSolver installation and basic functionality
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"PASS: PyTorch {torch.__version__}")
    except ImportError:
        print("FAIL: PyTorch not installed")
        return False
    
    try:
        import numpy as np
        print(f"PASS: NumPy {np.__version__}")
    except ImportError:
        print("FAIL: NumPy not installed")
        return False
    
    try:
        from src.model import ImprovedHyperGraphNet
        print("PASS: HyperSolver model")
    except ImportError as e:
        print(f"FAIL: HyperSolver model: {e}")
        return False
    
    try:
        from src.trainer import train_model
        print("PASS: HyperSolver trainer")
    except ImportError as e:
        print(f"FAIL: HyperSolver trainer: {e}")
        return False
    
    try:
        from src.data_reading import read_set_cover_instance
        print("PASS: Data reading functions")
    except ImportError as e:
        print(f"FAIL: Data reading: {e}")
        return False
    
    return True

def test_data_files():
    """Test that sample data files exist"""
    print("\nTesting data files...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("FAIL: Data directory not found")
        return False
    
    data_files = os.listdir(data_dir)
    if len(data_files) == 0:
        print("FAIL: No data files found")
        return False
    
    print(f"PASS: Found {len(data_files)} data files")
    return True

def test_configs():
    """Test that configuration files exist"""
    print("\nTesting configuration files...")
    
    config_dir = "configs"
    if not os.path.exists(config_dir):
        print("FAIL: Config directory not found")
        return False
    
    required_configs = [
        "set_cover_config.json",
        "subset_sum_config.json", 
        "hypermaxcut_config.json"
    ]
    
    for config in required_configs:
        if os.path.exists(os.path.join(config_dir, config)):
            print(f"PASS: {config}")
        else:
            print(f"FAIL: {config} missing")
            return False
    
    return True

def main():
    """Run all tests"""
    print("HyperSolver Setup Test\n")
    
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
        print("All tests passed! HyperSolver is ready to use.")
        print("\nTry running: python run.py --problem set_cover")
    else:
        print("Some tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
