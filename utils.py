import os
import random

def get_random_animation(folder="animations"):
    files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
    return os.path.join(folder, random.choice(files)) if files else None