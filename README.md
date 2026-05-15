# Parkinsons-Project-Final
TJHSST Senior Research Project - Diagnosing Parkinson's By Applying Graph Neural Networks to MRI

## Running surface extraction
Sign in to OSG access point (instructions here: https://osg-htc.org/services/access-point.html)
FreeSurfer is installed by default on the Open Science Grid; so, no installation necessary.
Upload PPMI_SCANS, freesurfer_run.sh, freesurfer.sub, and mvfiles.sh to OSG.
Run freesurfer.sub with 
'''<bash>
condor_submit freesurfer.sub

The resulting patient folders should appear in your working directory as '''.tar.gz files.
The folders are too large to move directly onto personal computer or cluster; so, first make a set of folders (recommended 3)
Then, use mvfiles.sh and shift each batch of folders into the specific folders you have created (just change "CORT_SURFACES2" to whichever folder name you created)
Around ~100 files may fail when extracting; this is normal

## Running graph creation & training graph neural network
After downloading the surface data, run the '''convert_scan_to_graph.py script by running the '''convert_graphs.sh :
'''<bash>
sbatch convert_graphs.sh

You may need to change the '''.../CORTSURFACES3 to whatever is the path of the folder(s) containing the patient scans you created
Create a directory titled '''PRELIM_GRAPHS before running the script
The script will additionally augment the graph dataset, and print the # of nodes and edges in each graph
A folder titled '''gcn_workspace should be created, where the script is also copied into

Training the graph neural network is straighforward - just run the '''train_gcn.sh script 
'''
sbatch train_gcn.sh

You may need to change the '''gcn_model1.py to the specific model you want to run inside the script
The script will also be copied to '''gcn_workspace and the best model(s) based on accuracy will also be saved inside of the gcn_workspace

## Running 3D model creation & training pre-trained 3D-CNN
To convert the .nii.gz scans to 3D volumes of the same orientation and dimensions, run '''convert_nii_to_3array.py via '''conversion.sh :
'''
sbatch conversion.sh

Create a directory titled '''3D_VOLUMES before running the script
Augment the dataset through '''data_augment_volume.py and '''volume_augment.sh :
'''
sbatch volume_augment.sh

Approximately ~3000 files should be created
A folder titled '''3dcnn_workspace should be created, where both scripts are copied into

Training the 3D-CNN is also fairly simple - just run the '''train_3dcnn.sh script :
'''
sbatch train_3dcnn.sh

The script will also be copied into '''3dcnn_workspace and the best model based on accuracy and 'checkpoint' models at each epoch will also be saved
