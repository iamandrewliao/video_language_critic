'''Give richer task descriptions to failures in raw-captions.pkl instead of just "Do nothing"'''

import pickle

def load_captions(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def extract_task_id(video_name):
    # Extracts the part after "__" and before the last "_"
    parts = video_name.split("__")
    if len(parts) > 1:
        task_and_idx = parts[1]
        idx = task_and_idx.rfind("_")  # rfind() returns the index of the last occurence of a substring
        return task_and_idx[:idx]
    return None

def save_captions(data, new_path):
    with open(new_path, 'wb') as f:
        pickle.dump(data, f)

captions_path = "/home/liao0241/video_language_critic/data/metaworld/mw40_split/raw-captions.pkl"
captions = load_captions(captions_path)

# Get a mapping from task name (task_id) to success task caption
caption_set = set()
success_map = {}

for vid, caption in captions.items():
    caption_tuple = tuple(caption[0])
    if vid.startswith("success_videos") and caption_tuple not in caption_set:
        caption_set.add(caption_tuple)
        task_id = extract_task_id(vid)
        assert task_id not in success_map  # if the caption is not in caption_set then task_id should also not be in success_map
        caption[0][0] = caption[0][0].lower() # change the first word in the caption to lowercase
        success_map[task_id] = caption[0]

# Relabel failure video captions
for vid in captions:
    if vid.startswith("fail_videos"):
        task_id = extract_task_id(vid)
        if task_id and task_id in success_map:
            new_caption = ['Fail', 'to'] + success_map[task_id]
            captions[vid] = [new_caption]

new_captions_path = "/home/liao0241/video_language_critic/data/metaworld/mw40_split/raw-captions-fails-relabeled.pkl"
save_captions(captions, new_captions_path)
print("Failure video captions relabeled.")