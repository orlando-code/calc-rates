import subprocess
from pathlib import Path

# get module working directory
def get_repo_root():
    # Run 'git rev-parse --show-toplevel' command to get the root directory of the Git repository
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if git_root.returncode == 0:
        return Path(git_root.stdout.strip())
    else:
        raise RuntimeError("Unable to determine Git repository root directory.")


# REPO DIRECTORIES
repo_dir = get_repo_root()
resources_dir = repo_dir / "resources"
meta_dir = repo_dir / "meta_2022"
module_dir = repo_dir / "calcification"
fig_dir = repo_dir / "figures"

# DATA DIRECTORIES
data_dir = repo_dir / "data"
climatology_data_dir = data_dir / "climatology"
tmp_data_dir = data_dir / "tmp"