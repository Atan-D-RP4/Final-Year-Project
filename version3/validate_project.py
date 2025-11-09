#!/usr/bin/env python3
"""Simple validation script for the financial forecasting project."""

import ast
import sys
from pathlib import Path


def validate_file_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path) as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    """Validate all Python files in the project."""
    project_root = Path(".")
    python_files = list(project_root.glob("**/*.py"))
    
    print(f"🔍 Validating {len(python_files)} Python files...")
    
    errors = []
    valid_files = []
    
    for file_path in python_files:
        is_valid, error = validate_file_syntax(file_path)
        if is_valid:
            valid_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            errors.append((file_path, error))
            print(f"❌ {file_path}: {error}")
    
    print(f"\n📊 Summary:")
    print(f"✅ Valid files: {len(valid_files)}/{len(python_files)}")
    print(f"❌ Files with errors: {len(errors)}")
    
    if errors:
        print("\n🔧 Files with syntax errors:")
        for file_path, error in errors:
            print(f"  {file_path}: {error}")
        return 1
    else:
        print("\n🎉 All Python files have valid syntax!")
        
        # Check key files exist
        key_files = [
            "src/models/baselines.py",
            "src/eval/metrics.py", 
            "src/analysis/__init__.py",
            "experiments/phase3/zero_shot.py",
            "tests/test_smoke.py",
            "src/utils/config.py",
            "src/utils/logger.py"
        ]
        
        missing_files = [f for f in key_files if not Path(f).exists()]
        if missing_files:
            print(f"\n⚠️  Missing key files: {missing_files}")
            return 1
        else:
            print("\n✅ All key files present!")
            
        # Check file sizes (basic complexity check)
        print(f"\n📏 File sizes (lines of code):")
        for file_path in key_files:
            if Path(file_path).exists():
                with open(file_path) as f:
                    lines = len(f.readlines())
                print(f"  {file_path}: {lines} lines")
        
        return 0


if __name__ == "__main__":
    sys.exit(main())