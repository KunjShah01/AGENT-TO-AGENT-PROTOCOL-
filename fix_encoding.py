#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix Unicode encoding issues in rla2a.py for Windows compatibility
"""

import sys
import os

def fix_unicode_issues():
    """Fix Unicode characters that cause encoding issues on Windows"""
    
    # Read the original file
    try:
        with open('rla2a.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("Error: rla2a.py not found")
        return False
    
    # Unicode character replacements for Windows compatibility
    replacements = {
        '🔍': '[CHECK]',
        '📦': '[INSTALL]', 
        '✅': '[OK]',
        '❌': '[FAIL]',
        '🚀': '[LAUNCH]',
        '🔐': '[SECURITY]',
        '🤖': '[AI]',
        '📚': '[DOCS]',
        '🛑': '[STOP]'
    }
    
    # Apply replacements
    for unicode_char, replacement in replacements.items():
        content = content.replace(unicode_char, replacement)
    
    # Add encoding declaration at the top if not present
    if not content.startswith('# -*- coding: utf-8 -*-'):
        content = '# -*- coding: utf-8 -*-\n' + content
    
    # Write the fixed content back
    try:
        with open('rla2a_fixed.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✓ Fixed file saved as rla2a_fixed.py")
        print("✓ All Unicode characters replaced with ASCII equivalents")
        print("✓ UTF-8 encoding declaration added")
        return True
    except Exception as e:
        print(f"Error writing fixed file: {e}")
        return False

if __name__ == "__main__":
    print("Fixing Unicode encoding issues...")
    if fix_unicode_issues():
        print("\nTo use the fixed version:")
        print("1. Backup your original: mv rla2a.py rla2a_original.py")
        print("2. Use the fixed version: mv rla2a_fixed.py rla2a.py")
        print("3. Run your command: python rla2a.py dashboard")
    else:
        print("Failed to fix encoding issues")
        sys.exit(1)