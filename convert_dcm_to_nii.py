import subprocess, os

dicom_folders = ["./PPMI/" + str(i) for i in range(3000, 236000)]
output_folder = "./PPMI_SCANS"

for folder in dicom_folders:
    if os.path.exists(folder):
        subprocess.run([
            "dcm2niix",
            "-z", "y", 
            "-o", output_folder, "-w", str(2),
            folder
    ])


