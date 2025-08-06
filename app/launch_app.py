#!/usr/bin/env python3
"""
Launcher script for Metafor Analysis Apps
Provides options to launch different versions of the app
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Interactive launcher for different app versions."""

    print("🦠 Metafor Analysis Dashboard Launcher")
    print("=" * 50)

    # Check environment
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

        # Available apps
    apps = {
        "1": {
            "name": "Ultimate App (RECOMMENDED)",
            "file": "metafor_ultimate_app.py",
            "description": "Best of all: Force R init + robust data cleaning + proper effect types",
        },
        "2": {
            "name": "Force Init R (Threading Fix)",
            "file": "metafor_force_init_app.py",
            "description": "Forces R initialization in main thread",
        },
        "3": {
            "name": "Robust App (Data Cleaning)",
            "file": "metafor_robust_app.py",
            "description": "Comprehensive data cleaning and error handling",
        },
        "4": {
            "name": "Context Manager R (Debug)",
            "file": "metafor_fixed_app.py",
            "description": "R context handling with detailed debugging",
        },
        "5": {
            "name": "Simple App (Fallback)",
            "file": "metafor_simple_app.py",
            "description": "Basic analysis with minimal R integration",
        },
        "6": {
            "name": "Original App (May have issues)",
            "file": "metafor_analysis_app.py",
            "description": "Original version with potential problems",
        },
    }

    print("\nAvailable apps:")
    for key, app in apps.items():
        status = "✅" if Path(app["file"]).exists() else "❌"
        print(f"{key}. {status} {app['name']}")
        print(f"   {app['description']}")

    print("\nWhich app would you like to launch?")
    choice = input("Enter choice (1-6): ").strip()

    if choice not in apps:
        print("❌ Invalid choice")
        sys.exit(1)

    app_file = apps[choice]["file"]
    app_path = Path(__file__).parent / app_file

    if not app_path.exists():
        print(f"❌ App file not found: {app_path}")
        sys.exit(1)

    print(f"\n🚀 Launching {apps[choice]['name']}...")
    print("📊 App will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")

    # Launch Streamlit
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
