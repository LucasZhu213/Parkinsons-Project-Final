# Parkinsons-Project-Final
TJHSST Senior Research Project - Diagnosing Parkinson's By Applying Graph Neural Networks to MRI

Parkinson’s Disease (PD) is a progressive neurodegenerative disorder and the second most common neurodegenerative disease in the elderly population.  While it is characterized by degradation of dopaminergic neurons in the Substantia nigra, it is primarily diagnosed by motor symptoms and ruling out contradictory symptoms; this method is fairly error-prone, especially for early diagnosis. Machine learning methods have been applied to medical imaging to diagnose Parkinson's; however, one biomarker not yet explored is regional cortical thinning, which has certain advantages over past methodologies, which focused on volumetrics. By extracting cortical surfaces from MRI, converting the geometric mesh to a graph, and applying Graph Neural Networks, we may improve early-stage discrimination. These models were compared against a logistic regression baseline trained on clinical evaluation data, as well as a combined multimodal approach integrating both models. The GNN-based models alone demonstrated limited predictive performance but showed relative sensitivity to intermediate disease stages. Logistic regression achieved strong overall classification performance, while the combined model improved class balance, particularly for prodromal cases. These results suggest that graph-based structural representations capture complementary information related to early disease progression but are insufficient as standalone predictors. This multimodal approach may support more informed clinical assessment and has potential implications for earlier detection and improved disease management, where earlier diagnosis is associated with improved long-term outcomes.

## Required packages & libraries
Python: Scipy, Scikit-Learn, Scikit-Image, Pandas, Numpy, MatPlotLib

Tensorflow, Keras, classification-models-3D

PyTorch, PyTorch Geometric

All are available on pip

This project is intended to be run on a cluster relying on SLURM and Pixi (for packages); if run elsewhere, will likely error.

## General Setup Instructions
In each file there are paths pointing to ```/csl/users/2026lzhu/...``` please change this to whatever is the **ABSOLUTE** path of the directory you have placed these in. ```convert_dcm_to_nii.py``` was used originally to convert the raw DiCOM files downloaded from PPMI to NiFTi format; the NiFTi dataset has been uploaded for your convenience. ```port_100.sh``` was used to upload the directory to GitHub.

## Running surface extraction
Sign in to OSG access point (instructions here: https://osg-htc.org/services/access-point.html). FreeSurfer is installed by default on the Open Science Grid; so, no installation necessary. Upload ```PPMI_SCANS```, ```freesurfer_run.sh```, ```freesurfer.sub```, ```clean_up_dir.sh``` and ```mvfiles.sh``` to OSG.

Run freesurfer.sub with 
```
condor_submit freesurfer.sub
```
Around ~100 files may fail when extracting; this is normal. The resulting patient folders should appear in your working directory as ```.tar.gz``` files. The folders are too large to move directly onto personal computer or cluster; so, first make a set of folders (recommended 3). Then, use mvfiles.sh and shift each batch of folders into the specific folders you have created (just change "CORT_SURFACES2" to whichever folder name you created). Then run ```clean_up_dir.sh``` (change "CORT_SURFACES" to the name of each of the folders you have created) to remove non-surface information and further reduce memory load.

## Training Logistic Regression
Run ```logreg.py``` to train the logistic regression model based on the MDS-UPDRS data (from the 3 csv files). Uncomment the section that merges, preprocesses and one-hot encodes the dataset. It should take a few minutes to run, and the final result will be saved in ```lr_model.joblib```. Before training ```gcn_model3.py```, you will need to run ```logreg.py```, and you may need to uncomment the section that trains the initial GCN+GraphNorm architecture. 

## Running graph creation & training graph neural network
After downloading the surface data, run the ```convert_scan_to_graph.py``` script by running the ```convert_graphs.sh```:

```
sbatch convert_graphs.sh
```
Create a directory titled ```PRELIM_GRAPHS``` before running the script. You may need to change the ```.../CORTSURFACES3``` to whatever is the path of the folder(s) containing the extracted surfaces you created. The script will additionally augment the graph dataset, and print the # of nodes and edges in each graph; around 2700-2800 files should be created. A folder titled ```gcn_workspace``` should be created, where the script is also copied into.

Training the graph neural network is straighforward - just run the ```train_gcn.sh``` script:
```
sbatch train_gcn.sh
```
You may need to change the ```gcn_model1.py``` to the specific model you want to run inside the script. The script will also be copied to ```gcn_workspace``` and the best model(s) based on accuracy will also be saved inside of ```gcn_workspace```, at ```best_graph_gcn_model#.pt```.

## Running 3D model creation & training pre-trained 3D-CNN
To convert the .nii.gz scans to 3D volumes of the same orientation and dimensions, run ```convert_nii_to_3array.py``` via ```conversion.sh```:
```
sbatch conversion.sh
```
Create a directory titled ```3D_VOLUMES``` before running the script. The script will additionally print the dimension of each scan (before resizing to same resolution). A folder titled ```3dcnn_workspace``` should be created, where the script should also be copied into.

Augment the dataset through ```data_augment_volume.py``` and ```volume_augment.sh```:
```
sbatch volume_augment.sh
```
Approximately ~3000 files should be created. The script will also be copied into ```3dcnn_workspace```.

Training the 3D-CNN is also fairly simple - just run the ```train_3dcnn.sh script```:
```
sbatch train_3dcnn.sh
```
The script will also be copied into ```3dcnn_workspace``` and the best model based on accuracy and 'checkpoint' models at each epoch will also be saved, in ```best_3d_cnn_model.keras``` and ```3d_cnn_temp_model.keras```, respectively.
