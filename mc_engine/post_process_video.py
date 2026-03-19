import os
import sys
import argparse
import subprocess
import shutil

# ── User data directory (same logic as queue_manager.py) ─────────────────────
def _user_data_dir() -> str:
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "MagicClick")
    elif sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "MagicClick")
    else:
        return os.path.join(os.path.expanduser("~"), ".magic_click")

_SHOTS_BASE = os.path.join(_user_data_dir(), "captured_shots")

# Import existing functions
from extract_frames import extract_frames  # type: ignore
from filter_scored_images import filter_scored_images  # type: ignore
try:
    from db_uploader import upload_best_frame  # type: ignore
    _DB_UPLOAD_ENABLED = True
except ImportError:
    _DB_UPLOAD_ENABLED = False

def main():
    parser = argparse.ArgumentParser(description="Post-process a recorded video: extract, score, and filter.")
    parser.add_argument("--video", required=True, help="Path to the recorded video .mp4 file")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: video {video_path} not found")
        return
        
    basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # 1. Extract Frames
    session_raw_dir = os.path.join(_SHOTS_BASE, "raw", f"{basename}_raw")
    print(f"\n[STEP 1] Extracting frames to {session_raw_dir}...")
    os.makedirs(session_raw_dir, exist_ok=True)
    extract_frames(video_path, session_raw_dir)
    
    # 2. Score Folder
    session_scored_dir = os.path.join(_SHOTS_BASE, "score", f"{basename}_scored")
    print(f"\n[STEP 2] Scoring frames to {session_scored_dir}...")
    os.makedirs(session_scored_dir, exist_ok=True)
    
    # Call score_folder.py via subprocess inside the mc_engine directory
    mc_engine_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable, os.path.join(mc_engine_dir, "score_folder.py"),
        "--input", session_raw_dir,
        "--output", session_scored_dir,
        "--no-viz"  # disable visualization images to save time and disk space
    ]
    
    ret = subprocess.run(cmd, cwd=mc_engine_dir)
    if ret.returncode != 0:
        print("\nError: Scoring process failed.")
        return
        
    # 3. Filter Images
    final_dir = os.path.join(_SHOTS_BASE, "final", f"{basename}_final")
    report_path = os.path.join(session_scored_dir, "results.json")
    print(f"\n[STEP 3] Filtering scored images to {final_dir}...")
    filter_scored_images(
        input_dir=session_scored_dir,
        source_dir=session_raw_dir,
        output_dir=final_dir,
        report_path=report_path
    )
    
    # 4. Clean up original tracked video
    print(f"\n[STEP 4] Deleting original video {video_path}...")
    try:
        os.remove(video_path)
    except Exception as e:
        print(f"Warning: Could not delete video {video_path}: {e}")
        
    # 5. Upload best frame to mc_database
    if _DB_UPLOAD_ENABLED:
        print(f"\n[STEP 5] Uploading best frame to mc_database...")
        try:
            upload_best_frame(session_scored_dir, basename)
        except Exception as e:
            print(f"[STEP 5] Upload error (non-fatal): {e}")
    else:
        print("\n[STEP 5] db_uploader not available — skipping DB upload.")

    print(f"\n[DONE] Post-processing for {video_path} complete. Best images saved to {final_dir}.")

if __name__ == "__main__":
    main()
