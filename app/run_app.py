#!/usr/bin/env python3
"""
Launch script for the Metafor Streamlit App
"""

import subprocess
import sys
from pathlib import Path


def check_environment():
    """Check if we're in the right environment with required packages."""
    required_packages = ["streamlit", "pandas", "plotly", "rpy2"]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Make sure you're in the 'calcer' conda environment:")
        print("   conda activate calcer")
        print("   pip install streamlit plotly")
        return False

    print("✅ All required packages found")
    return True


def main():
    """Launch the Streamlit app."""
    print("🔬 Launching Metafor Meta-Analysis Dashboard...")

    if not check_environment():
        sys.exit(1)

    # Get the app file path
    # app_file = Path(__file__).parent / "metafor_streamlit_app.py"
    app_file = Path(__file__).parent / "ui.py"

    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        sys.exit(1)

    print("📊 App will open in your browser at http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop the app")

    # Launch Streamlit
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_file),
                "--server.port",
                "8501",
                "--server.headless",
                "false",
            ]
        )
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
