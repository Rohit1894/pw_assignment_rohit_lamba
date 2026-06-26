import cv2
import os
import sys

def extract_frames(video_path, output_dir, prefix):
    if not os.path.exists(video_path):
        print(f"Error: {video_path} does not exist")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Video: {video_path}, FPS: {fps:.2f}, Frames: {total_frames}, Duration: {duration:.2f}s")
    
    # Extract frames at 10%, 30%, 50%, 70%, 90% of duration
    os.makedirs(output_dir, exist_ok=True)
    percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    for p in percentages:
        time_sec = duration * p
        frame_idx = int(time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_name = f"{prefix}_frame_{int(time_sec)}s.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, frame)
            print(f"Saved {out_path} at {time_sec:.2f}s")
        else:
            print(f"Failed to read frame at {time_sec:.2f}s")
            
    cap.release()

if __name__ == "__main__":
    output_dir = "e:/ai automation/pw_task_1/pw_assignment_rohit_lamba/output/analysis/extracted"
    
    # Extract from reference hindi_video.mp4
    ref_hindi = "e:/ai automation/pw_task_1/pw_assignment_rohit_lamba/input/vidos/hindi_video.mp4"
    extract_frames(ref_hindi, output_dir, "ref_hindi")
    
    # Extract from user's created hindi_v2.mp4
    user_hindi = "e:/ai automation/pw_task_1/pw_assignment_rohit_lamba/output/hindi_v2.mp4"
    extract_frames(user_hindi, output_dir, "user_hindi")
    
    # Extract from one math video in input/vidos to compare, e.g. 14.mp4 or 30.mp4
    math_video = "e:/ai automation/pw_task_1/pw_assignment_rohit_lamba/input/vidos/30.mp4"
    extract_frames(math_video, output_dir, "ref_math_30")
