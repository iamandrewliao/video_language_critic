'''Given a video, get a subsampled video or get the individual frames (raw, not transformed)'''

import os
import numpy as np
import cv2
from score_video import extract_frames

def get_subsample(og_video_path, max_frames, get_video=True):
    frames = extract_frames(og_video_path, max_frames=max_frames, stretch_partial=False)
    print(len(frames))
    video_name = os.path.splitext(os.path.basename(og_video_path))[0]
    if get_video:
        save_dir = f'video_language_critic/visualizations/{video_name}/video'
        os.makedirs(save_dir, exist_ok=True)
        height, width, _ = frames[0].shape
        fps = 12
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = os.path.join(save_dir, f'{video_name}_max_frames_{max_frames}.mp4')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        for i, img in enumerate(frames):
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255)
            img = img.astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)
        out.release()
        print(f"Video saved to {save_path} 🎥")
    else:
        save_dir = f'video_language_critic/visualizations/{video_name}/frames'
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(frames):
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255)
            img = img.astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(save_dir, f'{video_name}_max_frames_{max_frames}_frame{i}.png')
            cv2.imwrite(save_path, img_bgr)
        print(f"Frames saved to {save_dir} 🖼️")

MAX_FRAMES = 36
PATH = '/home/liao0241/video_language_critic/data/metaworld/mw50_videos/fail_videos_all_0.7_200_steps__basketball-v2_28.mp4'

get_subsample(PATH, MAX_FRAMES, get_video=False)