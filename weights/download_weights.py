import os
import shutil
import tempfile
import subprocess

WEIGHTS_REPO_URL = "https://github.com/Bariqa1/skin-condition-recognition-model-weights.git"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHT_FILES = [
    "efficientnet_s1_weights.pth",
    "efficientnet_s2_weights.pth"
]

def ensure_weights():
    """Checks if model weights exist locally; if not, downloads them from the GitHub weights repository."""
    missing = [f for f in WEIGHT_FILES if not os.path.exists(os.path.join(CURRENT_DIR, f))]
    if not missing:
        print("[✓] Model weights are already available in weights/ directory.")
        return True

    print(f"[*] Missing weights: {missing}. Downloading from repository...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        clone_cmd = ["git", "clone", "--depth=1", WEIGHTS_REPO_URL, tmp_dir]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[!] Error cloning weights repo: {result.stderr}")
            return False

        for f in WEIGHT_FILES:
            src = os.path.join(tmp_dir, f)
            dst = os.path.join(CURRENT_DIR, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"[✓] Downloaded and saved: {f}")
            else:
                print(f"[!] Warning: File {f} was not found in the cloned repository.")

    return True

if __name__ == "__main__":
    ensure_weights()
