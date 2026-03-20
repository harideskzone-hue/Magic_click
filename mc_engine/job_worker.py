import time
import subprocess
import sys
import os

sys.path.insert(0, os.getcwd())
import queue_manager  # type: ignore

def main():
    print("--- Starting Multi-Camera SJF Worker ---")
    print("Polling database 'captured_videos/jobs.db' for PENDING jobs...")
    
    while True:
        try:
            # Shortest Job First scheduling (SJF) is handled inherently by get_shortest_job()
            # which ORDER BY frame_count ASC
            job = queue_manager.get_shortest_job()
            
            if job is None:
                # No jobs, wait and poll again
                time.sleep(1.0)
                continue
            
            print("\n" + "═"*70)
            print(f" 🚀 [NEW JOB] Picking up session {os.path.basename(job['video_path'])}")
            print("═"*70)
            
            job_id = job['id']
            video_path = job['video_path']
            frame_count = job['frame_count']
            
            pending_count = queue_manager.get_pending_count()
            print(f"\n[JOB WORKER] Picked Job #{job_id} | Frames: {frame_count} | Mode: SJF")
            print(f"[JOB WORKER] Pending Jobs Remaining: {pending_count}")
            print(f"[JOB WORKER] Processing: {video_path}")
            
            # Execute the heavy ML pipeline strictly sequentially
            start_time = time.time()
            try:
                # We use the existing post_process_video.py logic for processing the mp4
                # NOTE: No capture_output so all logs (including DB uploads) print in real time
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "post_process_video.py")
                result = subprocess.run(
                    [sys.executable, script_path, "--video", video_path],
                    check=True
                )
                duration = time.time() - start_time
                print(f"[JOB WORKER] Job #{job_id} Completed in {duration:.1f}s.")
                queue_manager.mark_job_completed(job_id)

            except subprocess.CalledProcessError as e:
                print(f"[JOB WORKER] Job #{job_id} FAILED!")
                queue_manager.mark_job_failed(job_id)
                
        except KeyboardInterrupt:
            print("\nShutting down worker gracefully...")
            break
        except Exception as e:
            print(f"Worker Error: {e}")
            time.sleep(5.0)

if __name__ == "__main__":
    main()
