import tensorflow as tf, numpy as np, pandas as pd, nibabel as nib, matplotlib.pyplot as plt, math, re, os, random
import keras
from keras import layers
from scipy.ndimage import rotate, zoom
x = []
x_dir, cur_filename = "/csl/users/2026lzhu/3D_VOLUMES", 382637
subj_list = open("/csl/users/2026lzhu/scanlist.txt").readlines()
labels = pd.read_csv("LABELS.csv").set_index("Subject")
labels = labels[~labels.index.duplicated(keep='first')]
np.random.seed(42)
def augment(volume, max_rotation=10):
    if random.random() < 0.5:
        angle = np.random.uniform(-max_rotation, max_rotation)
        volume = rotate(volume, angle, axes=(1, 2), reshape=False, order=1, mode='nearest')
    if random.random() < 0.5:
        volume *= np.random.uniform(0.9, 1.1)
    volume += np.random.normal(0, 0.01, volume.shape)
    volume = (volume - np.mean(volume)) / (np.std(volume)+ 1e-8)
    return volume

random.seed(42)
already_seen = set()
for root, dirs, files in os.walk("/csl/users/2026lzhu/3D_VOLUMES"):
    for file in files:
        file = file.strip()
        subjnum = int(re.search("^((?!\_)\d)+", file).group())
        label = int(file[-5])
        if label in {0, 2}:
            if label == 2:
                image = np.load(f"/csl/users/2026lzhu/3D_VOLUMES/{subjnum}_2.npy")
                for i in range(1):
                    new_img, new_filename = augment(image), f"{subjnum}_{cur_filename+1}_2.npy"
                    np.save(os.path.join(x_dir, new_filename), new_img)
                    cur_filename += 1
            if label == 0:
                image = np.load(f"/csl/users/2026lzhu/3D_VOLUMES/{subjnum}_0.npy")
                for i in range(2):
                    new_img, new_filename = augment(image), f"{subjnum}_{cur_filename+1}_0.npy"
                    np.save(os.path.join(x_dir, new_filename), new_img)
                    cur_filename += 1
        else: 
            if random.random() < 1/9:
                image = np.load(f"/csl/users/2026lzhu/3D_VOLUMES/{subjnum}_1.npy")
                new_img, new_filename = augment(image), f"{subjnum}_{cur_filename+1}_0.npy"
                np.save(os.path.join(x_dir, new_filename), new_img)
                cur_filename += 1
    
    
    