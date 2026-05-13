import numpy as np, pandas as pd, nibabel as nib, math, re, os
from nibabel.processing import resample_to_output
from scipy.ndimage import zoom
subj_list = open("/csl/users/2026lzhu/scanlist.txt").readlines()
labels = pd.read_csv("/csl/users/2026lzhu/LABELS.csv").set_index("Subject")
labels = labels[~labels.index.duplicated(keep='first')]
target_shape = (96,128,128)
x_dir = "/csl/users/2026lzhu/3D_VOLUMES"
already_used = {3050:0}

for subj in subj_list:
    subj = subj.strip()
    subjnum = int(re.search("^((?!\_)\d)+", subj).group())
    if labels.loc[subjnum]["Group"] == "SWEDD": continue
    label = {"Control":0,"Prodromal":1,"PD":2}[labels.loc[subjnum]["Group"]]
    if subjnum in already_used: already_used[subjnum] += 1; subjnum = str(subjnum) + "abcdefghijklmnopqrstuvwxyz"[already_used[subjnum]-1]
    else: already_used[subjnum] = 0
    image = nib.load(f"/csl/users/2026lzhu/PPMI_SCANS/PPMI_SCANS/{subj}")
    image = nib.as_closest_canonical(image)
    if len(image.shape) != 3: continue
    image = resample_to_output(image, voxel_sizes=(1.0, 1.0, 1.0))
    img_data = image.get_fdata(dtype=np.float32)
    print(img_data.shape)
    factors = [t / s for t, s in zip(target_shape, img_data.shape)]
    img_data = zoom(img_data, factors, order=1)
    filename = f"{subjnum}_{label}.npy"
    np.save(os.path.join(x_dir, filename), img_data)


