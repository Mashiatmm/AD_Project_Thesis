# Environment Setup

`conda create ad_venv_2`

`conda activate ad_venv_2`

`conda install -c anaconda ipykernel`

`python -m ipykernel install --user --name=ad_venv_2`

`conda env export > environment.yml`


# Download from the existing environment of the repository :

`conda env create -f environment.yml`

Or update existing environment with the following command :

`conda activate ad_venv_2`

`conda env update --file environment.yml --prune` or

`conda env update --file environment.yml`


`--prune` uninstalls dependencies which were removed from .yml


# Errors

# Shap not showing any plot in jupyter

reassure `shap.initjs()` is there in the code and restart jupyter notebook (don't know why it works but it does )


# Neural Network Layer Explanation Link :

https://machinelearningknowledge.ai/different-types-of-keras-layers-explained-for-beginners/

# Neural Network Activation Layer Link :

https://towardsdatascience.com/7-popular-activation-functions-you-should-know-in-deep-learning-and-how-to-use-them-with-keras-and-27b4d838dfe6

check the gitignore file. find out the file system from there
in my modified file , you'll see folder name "PRSice_output" in place of "prsice_output" make modification ( preferably, change the folder name )
the ipynb with _modified_by_Naeem is the one I modified. this one should be followed

# SHAP for features:
https://docs.google.com/spreadsheets/d/1x-rmrGebP4Xtf_CfSzC3hbet-grB6WUsvURcc9FbWjI/edit#gid=0 

in place of reading from all_traits_map.json, we read from traits_map.json

# Papers

gwa tutorial: https://drive.google.com/file/d/1zxHoevqiou3DY3VXBe-GuLi5cUd3TPoF/view?usp=share_link

principal component analysis : https://sci-hub.se/https://doi.org/10.1002/gepi.22339 
