#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for rla2a.py to verify all functionality works correctly
"""

import sys
import subprocess
import importlib.util

def test_import():
    """Test if rla2a.py can be imported without errors"""
    print("[TEST] Testing import...")
    
    try:
        spec = importlib.util.spec_from_file_location("rla2a", "rla2a.py")
        rla2a = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rla2a)
        print("[OK] Import successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("[TEST] Testing basic functionality...")
    
    try:
        # Test help command
        result = subprocess.run([
            sys.executable, "rla2a.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("[OK] Help command works")
            return True
        else:
            print(f"[FAIL] Help command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        return False

def test_info_command():
    """Test info command"""
    print("[TEST] Testing info command...")
    
    try:
        result = subprocess.run([
            sys.executable, "rla2a.py", "info"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("[OK] Info command works")
            print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
            return True
        else:
            print(f"[FAIL] Info command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[FAIL] Info command test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("[AI] RL-A2A Test Suite")
    print("="*60)
    
    tests = [
        test_import,
        test_basic_functionality,
        test_info_command
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    print(f"[RESULT] {passed}/{total} tests passed")
    
    if passed == total:
        print("[OK] All tests passed! rla2a.py is working correctly.")
        print("[LAUNCH] You can now run: python rla2a.py dashboard")
    else:
        print("[FAIL] Some tests failed. Check the errors above.")
    
    print("="*60)

if __name__ == "__main__":
    main()