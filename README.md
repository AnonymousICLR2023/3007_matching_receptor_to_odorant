Data
----
https://drive.google.com/drive/folders/1Sb6nPaSgeX66Wo5uG8NYRmcMfvGr76_K?usp=sharing

Download data file: `Data_figures.zip` and `Dataset.zip` unzip them and place in the giving folder (create the folders if necessary):
  - Content of `Data` from `Data_figures.zip` to `Rec2Odorant/figures/Data`
  - Content of `Dataset` from `Dataset.zip` to `Rec2Odorant/Dataset`
  - Content of `_seq_dist_matrices` from `_seq_dist_matrices.zip` to `Rec2Odorant/Dataset/_seq_dist_matrices`

Preparation:
------------
After cloning the repo, several empty folders need to be created to save results:
```
cd <path_to_repo_3007_matching_receptor_to_odorant>
mkdir Rec2Odorant/figures/Figures
mkdir -p Rec2Odorant/main/CLS_GNN_MHA/Data/db
mkdir Rec2Odorant/main/CLS_GNN_MHA/logs
mkdir Rec2Odorant/odor_description/multiLabel/GNN/Data
mkdir Rec2Odorant/odor_description/multiLabel/GNN/logs
mkdir -p Rec2Odorant/odor_description/multiLabel/GNN/Prediction/embedding
```

Figures:
--------
In `figures` directiory, there is a jupyter notebook `plot_figures.ipynb` that generates all the figures as well as the full combinatorial code for all odor clusters. Some of the figures are saved to `Figures` folder.

All the necassary requirements except RDKit are in `figures/requirements.txt`.
To create the python environment run:
```
cd <path_to_figures_folder>
conda create -n <figures_env_name> python==3.9
conda activate <figures_env_name>
conda install -c conda-forge rdkit=2020.09.3
pip install -r requirements.txt
pip install -e <path_to_repo_3007_matching_receptor_to_odorant>
```
The jupyter notebook can be started by:
```
jupyter-notebook .
```

Main model:
-----------
There is a Singularity container prepared in `_container` with the environment to run the model. Singularity 3.7.0 and 3.8.0 was tested.
The container needs to be build from \_container/Rec2Odorant_singularity.def by running

```
cd <path_to_repo_3007_matching_receptor_to_odorant>/Rec2Odorant/_container
sudo singularity build Rec2Odorant_singularity.sif Rec2Odorant_singularity.def
```

This will create container `Rec2Odorant_singularity.sif` with all the python libraries installed.

### Start interactive session:
Huggingface needs a cache directory to download protBERT, but since the container doesn't allow writing after its creation, the default cache directory will be created in the host before starting the container. This directory is then mounted to the container.
```
mkdir -p <$HOME>/.cache/huggingface/transformers
```
```
singularity exec --containall --nv -B <path_to_repo_3007_matching_receptor_to_odorant>:/mnt,<$HOME>/.cache/huggingface/transformers,<other_folders_in_host_if_needed>:<other_folders_in_container> Rec2Odorant_singularity.sif bash
```
Inside the container run the following to be able to use conda and fix potential problem with cuda inside the container
```
LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
source /opt/miniconda3/etc/profile.d/conda.sh

conda activate base
pip install -e /mnt

cd /mnt/Rec2Odorant
```

### Split data and run main model:
Generate data splits (this will create 5 i.i.d. splits and 20 splits for generalization):
```
python main/datasets/db.py
```

Precompute protBERT [CLS] embedding:
```
python main/scripts/main_precompute.py --data_file=/mnt/Rec2Odorant/main/CLS_GNN_MHA/Data/db/<path_to_folder_containing_seqs_csv>/seqs.csv --save_dir=/mnt/Rec2Odorant/main/CLS_GNN_MHA/Data/db/<path_to_folder_containing_seqs_csv> --cuda_device=0
```

Run the model:
```
python main/scripts/main.py --config=/mnt/Rec2Odorant/main/configs/localhost/config_train_example.yml --cuda_device=0
```
The model can be run using config files. There are few examples of config files in main/configs. Change the config file to run other models. The ACTION at the beginning of the config file changes the type of action to be performed. This can be either `train`, `eval`, `predict` or `predict_single`. Note that config_train_example.yml will not work as is because the path to data splits will change.

Odor embedding:
---------------
### Data:
Since the `/opt` folder in the container is not writable, the data needs to be downloaded outside of the container.
To get the data, **ouside of the container** use conda environment with `pyrfume` and `Rec2Odorant` (this library) installed (for example `conda activate <figures_env_name>` create above) and run 
```
cd <path_to_repo_3007_matching_receptor_to_odorant>/Rec2Odorant
python odor_description/multiLabel/GNN/datasets/pyrfume/main.py
```
To run the training script, start the container if you did not already do so (see subsection Interactive session) and similarily to the main model, 
change the config file in `Rec2Odorant/odor_description/configs/localhost` and run
```
python /mnt/Rec2Odorant/odor_description/scripts/main.py --config=/mnt/Rec2Odorant/odor_description/configs/localhost/config_train.yml
```
To get the embedding just modify the config and run the same script with different config:
```
python /mnt/Rec2Odorant/odor_description/scripts/main.py --config=/mnt/Rec2Odorant/odor_description/configs/localhost/config_predict.yml
```
