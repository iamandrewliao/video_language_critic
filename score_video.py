'''
This script scores arbitrary videos with the VLC reward function.
'''
import argparse
import os
import pickle
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # need this to import video_language_critic.reward
from video_language_critic.reward import RewardCalculator
# from vlc_rl.paths import *
import numpy as np
from PIL import Image
import torch
import cv2

# Load VLC args (helper)
def load_vlc_args(vlc_ckpt):
    '''Load VLC model arguments from pkl file'''
    # make sure your config file is in the same folder as the model, and with this naming format!
    vlc_args_path = os.path.join(vlc_ckpt + '_config.pkl')
    with open(vlc_args_path, 'rb') as f:
        vlc_args = pickle.load(f)['args']  # just load the arguments
    vlc_args.init_model = vlc_ckpt
    vlc_args.resume_from_latest = False
    return vlc_args

# Parse arguments (helper)
def parse_args():
    parser = argparse.ArgumentParser(description="Score videos using VLC reward model")
    parser.add_argument("--vlc-ckpt", type=str, required=True, help="Path to VLC checkpoint under paths/REWARD_CKPT_DIR")
    parser.add_argument("--video-path", type=str, required=True, help="Path to video file or directory of videos")
    parser.add_argument("--caption", type=str, help="Custom description for the task (not needed)")
    parser.add_argument("--stretch-partial", action="store_true", help="If true, stretch partial videos to fill max frames")
    # action="store_true" means that if it is included in the command line, it will be True otherwise False
    # default=True means the value will default to True even if not included in command line
    parser.add_argument("--reward-normalization-offset", action="store_true", default=True, help="Shift rewards so first frame has reward 0")  
    # parser.add_argument("--reward-normalization-gymnasium", action="store_true", default=True,
    #                     help="Apply Gymnasium reward normalization")  # same here
    # parser.add_argument("--gamma", type=float, default=0.99,
    #                     help="Discount factor for reward normalization")
    # parser.add_argument("--output-file", type=str, default=None, help="Save results to output file")
    parser.add_argument("--fails-relabeled", action="store_true", help="If specified, using captions file with fails relabeled into more useful captions rather than raw-captions.pkl")
    parser.add_argument("--visualize", action="store_true", help="If specified, visualize reward")
    return parser.parse_args()

# extract frames from video (helper)
def extract_frames(video_path, max_frames, stretch_partial):
    """Extract frames from a video file  
    max_frames and stretch_partial come from args
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames >= max_frames or stretch_partial:
        # evenly sample across video
        # even if video is too short, linspace takes care of "stretching" partial videos, repeating some indices)
        indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)
    elif num_frames < max_frames or not stretch_partial:
        # if the videos are too short and we aren't stretching, just give an evenly spaced range across the video and pad the rest
        indices = np.arange(num_frames + 1)  # [0, 1,..., num_frames]
    indices_set = set(indices.tolist())

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count in indices_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        count += 1
    cap.release()
    return frames

# Transform video (helper)
def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=Image.Resampling.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
)

# Prepare text for scoring (helper)
def process_caption(reward_model, caption):
    """
    Process caption text to get necessary tensors: pairs_text, pairs_mask, pairs_segment
    """
    # Access the dataloader's tokenizer and configurations
    dataloader = reward_model.dataloader
    max_words = dataloader.max_words
    tokenizer = dataloader.tokenizer
    special_tokens = dataloader.SPECIAL_TOKEN
    
    # Process the caption (similar to _get_text() in VLM_DataLoader() from video-language-critic repo)
    k = 1  # We're processing one caption at a time
    pairs_text = np.zeros((k, max_words), dtype=np.compat.long)
    pairs_mask = np.zeros((k, max_words), dtype=np.compat.long)
    pairs_segment = np.zeros((k, max_words), dtype=np.compat.long)

    # Process the caption
    words = tokenizer.tokenize(caption)
    
    # Add special tokens
    words = [special_tokens["CLS_TOKEN"]] + words
    total_length_with_CLS = max_words - 1
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [special_tokens["SEP_TOKEN"]]

    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    
    # Pad to max_words
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    # Store in arrays
    pairs_text[0] = np.array(input_ids)
    pairs_mask[0] = np.array(input_mask)
    pairs_segment[0] = np.array(segment_ids)
    
    return pairs_text, pairs_mask, pairs_segment

# Score video (helper)
def score_video_frames(reward_model, video_frames, caption, max_frames, normalize_offset, loss_type):
    """Output raw and normalized VLC reward history for video frames and caption
    video_frames come from extract_frames()"""
    # Process frames and convert to tensors
    transform = _transform(224)  # CLIP resolution
    processed_frames = torch.zeros((len(video_frames), 3, 224, 224))
    for i, frame in enumerate(video_frames):
        processed_frames[i] = transform(Image.fromarray(frame))

    # Prepare text input
    pairs_text, pairs_mask, pairs_segment = process_caption(reward_model, caption)
    pairs_text = torch.from_numpy(np.asarray(pairs_text)).to(DEVICE)
    pairs_mask = torch.from_numpy(np.asarray(pairs_mask)).to(DEVICE)
    pairs_segment = torch.from_numpy(np.asarray(pairs_segment)).to(DEVICE)
    
    # Collect scores for each frame
    reward_history = []
    
    with torch.no_grad():
        # At every timestep, create batch with frames up to current frame
        for i in range(1, len(video_frames) + 1):
            # Select frames up to current frame
            curr_frames = processed_frames[:i]  # torch.Size([i, 3, 224, 224])

            num_frames = curr_frames.shape[0]  # should be = i
            # if video is too short (stretch_partial=False), pad the rest
            # otherwise, if video at this point in the loop is max_frames long or stretch_partial=True, num_frames will = max_frames and num_padded will = 0
            num_padded = max_frames - num_frames

            video_mask = torch.from_numpy(np.asarray([1] * num_frames + [0] * num_padded)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Create batch with shape [num_env (a sac_jax.py thing, irrelevant here), 1, max_frames, 1, 3, 224, 224]
            batch = torch.zeros((1, 1, max_frames, 1, 3, 224, 224))
            batch[0, 0, :num_frames] = curr_frames.unsqueeze(1).unsqueeze(0)
            # rest of the frames batch[0][0][num_frames:max_frames] will be padded with zeros
            # print(batch.shape)
            # print(batch)
            # Get similiarity scores
            a, b = reward_model.model.get_sequence_visual_output(pairs_text, pairs_mask, pairs_segment, batch.to(DEVICE), video_mask)
            # print(a.shape, b.shape)
            # a: torch.Size([1, 1, 512]), b: torch.Size([1, 12, 512]);
            scores = reward_model.model.get_similarity_logits(a, b, pairs_text, video_mask, loose_type=reward_model.model.loose_type)[0]
            # scores: torch.Size([1, 1, 12]) for sequence_ranking_loss, 
            # scores: torch.Size([1, 1]) for cross_entropy, goal_similarity_loss
            # the difference is due to self.return_sequence
            # Get the current window's score S(v_1:i,c)
            # print(scores.shape, scores[0])
            if loss_type == 'sequence_ranking_loss' or loss_type == 'goal_sequence_ranking_loss':
                curr_score = scores[0][0][i-1]
            else:
                curr_score = scores[0]
            reward_history.append(curr_score.cpu().numpy())
    
    reward_history = np.array(reward_history)

    if normalize_offset and len(reward_history) > 0:  # offset normalization
        normalized_reward_history = np.zeros(len(reward_history))
        normalized_reward_history = reward_history-reward_history[0]
    else:
        normalized_reward_history = None

    return reward_history, normalized_reward_history

# Visualize rewards
def visualize_reward(video_path, scores, save_path, vlc_ckpt, just_normalized=True, save=True):
    """Create a visualization of scores over frames  
    Args:  
    just_normalized: only normalized scores or raw and normalized
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        if just_normalized:
            plt.plot(range(1,len(scores['normalized_scores'])+1), scores['normalized_scores'], label='Normalized')
        else:
            plt.plot(range(1,len(scores['raw_scores'])+1), scores['raw_scores'], label='Raw')
            plt.plot(range(1,len(scores['normalized_scores'])+1), scores['normalized_scores'], label='Normalized')
        plt.xlabel('Frame')
        plt.ylabel('Score')
        plt.title(f'VLC Reward: {os.path.basename(video_path)}\n{vlc_ckpt}')
        plt.legend()
        plt.grid(True)
        if save:
            if just_normalized:
                plt.savefig(f'{save_path}/normalized_only.png')
            else:
                plt.savefig(f'{save_path}/raw_and_normalized.png')
            print(f"Reward plot saved to {save_path} 📈")
        else:
            plt.show()
    except ImportError:
        print("Failed matplotlib import")

# Main function
def main():
    # Load args and reward model
    args = parse_args()
    vlc_args = load_vlc_args(args.vlc_ckpt)
    reward_model = RewardCalculator(args=vlc_args)
    reward_model.model.eval()
    
    # Process videos
    if os.path.isdir(args.video_path):
        video_paths = [os.path.join(args.video_path, f) for f in os.listdir(args.video_path)
                       if f.endswith(('.mp4', '.avi', '.mov'))]
    else:
        video_paths = [args.video_path]
    
    results = {}
    for video_path in video_paths:
        print(f"Processing {video_path}...")
        video_name = os.path.basename(video_path)
        video_name_without_extension = os.path.splitext(video_name)[0]
        try:
            # Extract frames
            frames = extract_frames(video_path, max_frames=vlc_args.max_frames, stretch_partial=args.stretch_partial)
            if not frames:
                raise ValueError(f"No frames extracted from {video_path}")
            print(f"Extracted {len(frames)} frames")
            # Score video frames and apply reward normalization
            if args.caption:  # if custom caption provided, use it
                raw_scores, normalized_scores = score_video_frames(reward_model, frames, args.caption, vlc_args.max_frames, args.reward_normalization_offset, vlc_args.loss_type)
            else:  # otherwise, just get caption from provided .pkl file
                if args.fails_relabeled:
                    captions_path = '/home/liao0241/video_language_critic/data/metaworld/mw40_split/raw-captions-fails-relabeled.pkl'
                else:
                    captions_path = '/home/liao0241/video_language_critic/data/metaworld/mw40_split/raw-captions.pkl'
                with open(captions_path, 'rb') as f:
                    data = pickle.load(f)  # read captions
                    if isinstance(data, dict):
                        caption = " ".join(data[video_name_without_extension][0])  # get caption and reformat as single string
                        raw_scores, normalized_scores = score_video_frames(reward_model, frames, caption, vlc_args.max_frames, args.reward_normalization_offset, vlc_args.loss_type)
                    else:
                        raise TypeError('Pickle data should be a dict')
            results[video_name] = {'raw_scores': raw_scores, 'normalized_scores': normalized_scores}
            print(f'Video: {video_name}\nRaw scores: {raw_scores}\nNormalized scores: {normalized_scores}')

            # Visualize if desired
            if args.visualize:
                vlc_ckpt_name = os.path.basename(args.vlc_ckpt)
                save_dir = f'visualizations/{video_name_without_extension}/{vlc_ckpt_name}/reward_plots'
                os.makedirs(save_dir, exist_ok=True)
                visualize_reward(video_path=video_path, 
                                 scores=results[video_name], 
                                 save_path=save_dir, 
                                 vlc_ckpt=vlc_ckpt_name,
                                 just_normalized=True,
                                 save=True)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    return results

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # everybody always uses 0
print(f"Using device: {DEVICE}")

if __name__ == "__main__":
    main()

# Example run commands: 
# mw40 ckpt: python score_video.py --vlc-ckpt /home/liao0241/video_language_critic/vlc_rl/REWARD_CKPT/vlc_ckpts/ckpt_mw40_retrank33_tigt_negonly_a_rf_1__pytorch_model.bin.20 --video-path /home/liao0241/video_language_critic/data/metaworld/mw50_videos/success_videos__basketball-v2_48.mp4 --reward-normalization-offset --visualize
# mw50 ckpt: python score_video.py --vlc-ckpt /home/liao0241/video_language_critic/vlc_rl/REWARD_CKPT/vlc_ckpts/ckpt_mw50_retrank33_tigt_negonly_a_rf_1__pytorch_model.bin.20 --video-path /home/liao0241/video_language_critic/data/metaworld/mw50_videos/success_videos__basketball-v2_48.mp4 --reward-normalization-offset --visualize
# openx ckpt: python score_video.py --vlc-ckpt /home/liao0241/video_language_critic/vlc_rl/REWARD_CKPT/vlc_ckpts/ckpt_openx_retrieval_tigt_negonly_a_rf_1__pytorch_model.bin.15 --video-path /home/liao0241/video_language_critic/data/metaworld/mw50_videos/success_videos__basketball-v2_48.mp4 --reward-normalization-offset --visualize
