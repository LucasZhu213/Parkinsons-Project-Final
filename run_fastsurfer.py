import subprocess, os, re, shutil

paths = open("scanlist.txt").read().splitlines()
paths = [p.strip() for p in paths]
uid = os.getuid()
gid = os.getgid()

base_dir = os.getcwd()
license_path = os.path.abspath("./freesurfer/license.txt")
subjects_dir = os.path.abspath("./CORT_SURFACES")
already_used = {3050:0, 100001:0}

for filename in paths:
    subjnum = int(re.search("^((?!\_)\d)+", filename).group())
    if subjnum in already_used: already_used[subjnum] += 1; subjnum = str(subjnum) + "abcdefghijklmnopqrstuvwxyz"[already_used[subjnum]-1]
    else: already_used[subjnum] = 0
    t1_path = os.path.abspath(f"./PPMI_SCANS/{filename}")
    subprocess.run([
        os.path.join(base_dir, "FastSurfer", "run_fastsurfer.sh"), "--fs_license", license_path, "--t1", t1_path, "--sid", "subject"+str(subjnum),  "--sd", subjects_dir, "--3T", "--threads", "4",
        "--parallel", "--device", "auto"
    ], check=True)
            
