"""
Force R initialization in main thread before Streamlit imports
This should be imported before any Streamlit code to ensure proper R context
"""

import threading
import warnings

# Suppress R warnings early
warnings.filterwarnings("ignore", message="R is not initialized by the main thread")
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

print(f"[R_INIT] Initializing R in thread: {threading.current_thread().name}")
print(
    f"[R_INIT] Is main thread: {threading.current_thread() is threading.main_thread()}"
)

# Force R initialization in main thread
try:
    # Import and activate rpy2 immediately
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    print("[R_INIT] rpy2 modules imported successfully")

    # Activate pandas conversion globally
    pandas2ri.activate()
    print("[R_INIT] pandas2ri activated globally")

    # Import R packages
    metafor = importr("metafor")
    base = importr("base")
    print("[R_INIT] R packages imported successfully")

    # Test conversion
    import pandas as pd

    test_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_test = ro.conversion.py2rpy(test_df)
        print("[R_INIT] Test conversion successful")

    # Store in global variables for reuse
    globals()["_R_INITIALIZED"] = True
    globals()["_METAFOR"] = metafor
    globals()["_BASE"] = base
    globals()["_RO"] = ro
    globals()["_PANDAS2RI"] = pandas2ri
    globals()["_LOCALCONVERTER"] = localconverter

    print("[R_INIT] R initialization complete and stored globally")

except Exception as e:
    print(f"[R_INIT] R initialization failed: {e}")
    import traceback

    print(f"[R_INIT] Traceback: {traceback.format_exc()}")
    globals()["_R_INITIALIZED"] = False


def get_r_objects():
    """Get the initialized R objects."""
    if globals().get("_R_INITIALIZED", False):
        return {
            "metafor": globals()["_METAFOR"],
            "base": globals()["_BASE"],
            "ro": globals()["_RO"],
            "pandas2ri": globals()["_PANDAS2RI"],
            "localconverter": globals()["_LOCALCONVERTER"],
        }
    else:
        return None


def is_r_initialized():
    """Check if R was successfully initialized."""
    return globals().get("_R_INITIALIZED", False)
