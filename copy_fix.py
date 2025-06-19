#!/usr/bin/env python3
"""
Quick fix script - copies the fixed version to rla2a.py
"""
import shutil
import os

# Copy the fixed file to rla2a.py
if os.path.exists('rla2a_complete_fixed.py'):
    shutil.copy('rla2a_complete_fixed.py', 'rla2a.py')
    print("[OK] Fixed file copied to rla2a.py")
    print("[LAUNCH] You can now run: python rla2a.py dashboard")
else:
    print("[FAIL] rla2a_complete_fixed.py not found")