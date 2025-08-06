#!/usr/bin/env python3
"""
Launcher script for the Metafor Analysis App
"""

import subprocess
import sys
from pathlib import Path

# def check_requirements():
#     """Check if required packages are installed."""
#     try:
#         import plotly
#         import streamlit

#         print("✅ Required packages found")
#         return True
#     except ImportError as e:
#         print(f"❌ Missing required package: {e}")
#         print("Install with: pip install -r requirements_app.txt")
#         return False


def main():
    """Launch the Streamlit app."""
    # Check if we're in the right environment
    try:
        import plotly
        import streamlit

        print("✅ Required packages found")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Make sure you're in the 'calcer' conda environment:")
        print("   conda activate calcer")
        print("   pip install -r requirements_app.txt")
        sys.exit(1)

    # Try the ultimate app first (combines best of all approaches)
    app_paths = [
        Path(__file__).parent / "metafor_ultimate_app.py",
        Path(__file__).parent / "metafor_force_init_app.py",
        Path(__file__).parent / "metafor_robust_app.py",
        Path(__file__).parent / "metafor_simple_app.py",
    ]

    app_path = None
    for path in app_paths:
        if path.exists():
            app_path = path
            break

    if not app_path:
        print(f"❌ No app files found. Tried: {[str(p) for p in app_paths]}")
        sys.exit(1)

    print("🚀 Launching Metafor Analysis Dashboard...")
    print("📊 App will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")

    # Launch Streamlit on a different port to avoid conflicts
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            "8501",
            "--browser.gatherUsageStats",
            "false",
        ]
    )


if __name__ == "__main__":
    main()
