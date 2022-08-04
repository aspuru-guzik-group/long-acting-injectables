# long-acting-injectables

Data and code used for the resaerch article titled "Machine Learning Models to Accelerate the Design of Polymeric Long-Acting Injectables".

The model directories are named for the machine learning approcah deployed within. Each model directory contains two python scripts that can be used to train and validate the respective machine learning models using either leave-one-group-out cross-validation (LOGO_CV) or NESTED cross-validation (NESTED_CV). The "RandomForest" directory also contains the code used to refine the LOGO_CV RF model from 14-input features to 12-input features, as well as a pickle file (.pkl) conatining the saved hyperparamaters for the final version fo this 12-feature RF mdoel described in the aformentioned study.

The "DATA" directory conatins the raw datasets nescessary to train and validate the models using the aformentioned code. The "DATA" directory is subdivided into LOGO_CV and NESTED_CV. The LOGO_CV subdirectory contains the training and validation datasets used in this study for the LOGO_CV models. Both of these datasets are provided as xlsx, csv, and tsv file types. The NESTED_CV subdirectory contains the single dataset (essentially the aformentioned training and validation datasets combined) that was used in this study for NESTED_CV models. This dataset is provided as xlsx, csv, and tsv file types.

Link to preprint: https://doi.org/10.26434/chemrxiv-2021-mxrxw-v2
