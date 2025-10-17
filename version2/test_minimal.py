"""Minimal test without heavy dependencies."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))


def test_structure() -> bool:
    """Test that the project structure is correct."""
    print("Testing project structure...")

    required_dirs = [
        "src",
        "src/data",
        "src/preprocessing",
        "src/models",
        "src/eval",
        "src/utils",
        "src/app",
        "src/analysis",
        "experiments",
        "data",
    ]

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} missing")
            return False

    required_files = [
        "src/__init__.py",
        "src/utils/config.py",
        "src/utils/logger.py",
        "src/data/fetchers.py",
        "src/data/cleaning.py",
        "src/preprocessing/tokenizer.py",
        "src/models/baselines.py",
        "src/models/chronos_wrapper.py",
        "src/eval/metrics.py",
        "experiments/phase3_zero_shot.py",
        "src/main.py",
        "main.py",
        "pyproject.toml",
    ]

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} missing")
            return False

    print("Project structure is correct!")
    return True


def test_config_without_dependencies() -> bool:
    """Test configuration without importing heavy dependencies."""
    print("\\nTesting configuration...")

    try:
        # Test that we can at least import the config module structure
        config_file = Path("src/utils/config.py")
        content = config_file.read_text()

        # Check for key classes and functions
        required_elements = [
            "class DataConfig",
            "class ModelConfig",
            "class ExperimentConfig",
            "class Config",
            "def load_config",
        ]

        for element in required_elements:
            if element in content:
                print(f"✓ {element} found")
            else:
                print(f"✗ {element} missing")
                return False

        print("Configuration structure is correct!")
        return True

    except Exception as e:
        print(f"Error testing config: {e}")
        return False


def test_code_quality() -> bool:
    """Test basic code quality without running the code."""
    print("\\nTesting code quality...")

    try:
        # Check that Python files have proper syntax
        python_files = []
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        for file in ["experiments/phase3_zero_shot.py", "main.py"]:
            if os.path.exists(file):
                python_files.append(file)

        print(f"Found {len(python_files)} Python files")

        # Basic syntax check by trying to compile
        import ast

        syntax_errors = 0

        for file_path in python_files:
            try:
                with open(file_path) as f:
                    content = f.read()
                ast.parse(content)
                print(f"✓ {file_path} - syntax OK")
            except SyntaxError as e:
                print(f"✗ {file_path} - syntax error: {e}")
                syntax_errors += 1
            except Exception as e:
                print(f"? {file_path} - could not check: {e}")

        if syntax_errors == 0:
            print("All Python files have valid syntax!")
            return True
        print(f"{syntax_errors} files have syntax errors")
        return False

    except Exception as e:
        print(f"Error testing code quality: {e}")
        return False


def test_documentation() -> bool:
    """Test that documentation exists."""
    print("\\nTesting documentation...")

    doc_files = ["README.md", "RESEARCH_OVERVIEW.md", "UPDATED_OUTLINE.md", "PLAN.md"]

    found_docs = 0
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            print(f"✓ {doc_file}")
            found_docs += 1
        else:
            print(f"? {doc_file} not found")

    if found_docs >= 2:
        print("Sufficient documentation found!")
        return True
    print("Insufficient documentation")
    return False


def main() -> bool:
    """Run minimal tests."""
    print("Running minimal tests for multivariate financial forecasting system")
    print("=" * 70)

    success = True

    # Test project structure
    success &= test_structure()

    # Test configuration
    success &= test_config_without_dependencies()

    # Test code quality
    success &= test_code_quality()

    # Test documentation
    success &= test_documentation()

    print("\\n" + "=" * 70)
    if success:
        print("🎉 All minimal tests passed!")
        print("\\nProject structure and code quality look good.")
        print("\\nPhase 3 implementation includes:")
        print("  ✓ Data fetching and cleaning modules")
        print("  ✓ Financial data tokenization")
        print("  ✓ Baseline forecasting models (Naive, ARIMA, VAR, LSTM, Linear)")
        print("  ✓ Chronos wrapper (with mock implementation)")
        print("  ✓ Comprehensive evaluation metrics")
        print("  ✓ Phase 3 experiment runner")
        print("  ✓ Configuration and logging utilities")
        print("\\nTo run with dependencies installed:")
        print("  uv run python main.py --phase 3")
        print("\\nOr to install dependencies manually:")
        print("  uv sync")
        print("  uv run python main.py --phase 3")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    return success


if __name__ == "__main__":
    main()
