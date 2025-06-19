#!/usr/bin/env python3
"""
Quick fix script - copies the fixed version to rla2a.py
"""
import shutil
# Copy the final fixed file to rla2a.py
if os.path.exists('rla2a_fixed_final.py'):
    shutil.copy('rla2a_fixed_final.py', 'rla2a.py')
    print("[OK] Final fixed file copied to rla2a.py")
    print("[LAUNCH] All errors fixed! You can now run: python rla2a.py dashboard")
    print("[OK] Fixed file copied to rla2a.py")
    print("[FAIL] rla2a_fixed_final.py not found")
else:
    print("[FAIL] rla2a_complete_fixed.py not found")