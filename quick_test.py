#!/usr/bin/env python3
"""
Quick test script to verify HyperSolver works with the repository structure
"""

import os
import sys

def test_data_files():
    """Check that data files exist"""
    print("Checking data files...")
    data_dir = "./data"
    if not os.path.exists(data_dir):
        print(f"FAIL: {data_dir} directory not found")
        return False
    
    files = os.listdir(data_dir)
    if len(files) == 0:
        print(f"FAIL: No files in {data_dir}")
        return False
    
    print(f"PASS: Found {len(files)} files in {data_dir}")
    
    # Check specific files
    subset_sum_files = [f for f in files if f.startswith('final_')]
    hypergraph_files = [f for f in files if f.startswith('Hyp_')]
    
    print(f"  - Subset sum files: {len(subset_sum_files)}")
    print(f"  - Hypergraph files: {len(hypergraph_files)}")
    
    return True

def test_configs():
    """Check that config files have correct paths"""
    print("\nChecking config files...")
    config_dir = "./configs"
    
    if not os.path.exists(config_dir):
        print(f"FAIL: {config_dir} directory not found")
        return False
    
    import json
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
    
    for config_file in config_files[:5]:  # Test first 5 configs
        try:
            with open(os.path.join(config_dir, config_file), 'r') as f:
                config = json.load(f)
                
            if 'folder_path' in config:
                path = config['folder_path']
                if path.startswith('/Users/'):
                    print(f"FAIL: {config_file} still has hardcoded path: {path}")
                    return False
                else:
                    print(f"PASS: {config_file} uses relative path: {path}")
        except Exception as e:
            print(f"WARN: Could not parse {config_file}: {e}")
    
    return True

def test_imports():
    """Test that core modules can be imported"""
    print("\nTesting imports...")
    
    try:
        from src.model import ImprovedHyperGraphNet
        print("PASS: Model import successful")
    except ImportError as e:
        print(f"FAIL: Model import failed: {e}")
        return False
    
    try:
        from src.data_reading import read_set_cover_instance
        print("PASS: Data reading import successful")
    except ImportError as e:
        print(f"FAIL: Data reading import failed: {e}")
        return False
    
    try:
        from src.trainer import train_model
        print("PASS: Trainer import successful")
    except ImportError as e:
        print(f"FAIL: Trainer import failed: {e}")
        return False
    
    return True

def main():
    print("HyperSolver Repository Test\n")
    print("="*50)
    
    all_passed = True
    
    if not test_data_files():
        all_passed = False
    
    if not test_configs():
        all_passed = False
    
    if not test_imports():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("All tests passed! Repository is ready for use.")
        print("\nNext steps:")
        print("1. Run: python run.py --problem subset_sum")
        print("2. Run: python run.py --problem set_cover") 
        print("3. See TESTING_GUIDE.md for detailed instructions")
    else:
        print("Some tests failed. Check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
