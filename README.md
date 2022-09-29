# long-acting-injectables

Data and code used for the research article titled "Machine Learning Models to Accelerate the Design of Polymeric Long-Acting Injectables".

There are directories for (i) few-shot models and (ii) zero-shot models.

The few-shot models directory conatins the datafile used to train these models (Dataset_17_feat) as various files types (xlsx, csv, and tsv).
The zero-shot models directory conatins the datafile used to train these models (Dataset_14_feat) as various files types (xlsx, csv, and tsv).

Each directory contains a python class (NESTED_CV_) that is called for the nested cross-validation of either the few-shot or zero-shot machine learning models. This class is called to train all of the machine learning models in this study (except for the neural networks). Once implemented, a 10-fold nested cross-validation is conducted on the specified model. The results of this nested cross-validation are stored in a sub-directory (NESTED_CV_RESULTS) as a pickle file (.pkl). The best model hyperparameter configuration is stored in a sub-directory (Trained_models) as a pickle file (.pkl). 

Each directory also contains the python scripts used for (i) preliminary model evaluation, (ii) refinement and re-training of the "best" model, and (iii) to call the trained model to make predictions.

Each directory also contains all of the codes necessary to replicate the figures used in the research article (e.g., Figure_1, Figure_2, etc.). There is an additional sub-directory (Figures) that stores all of the figures generated using these python scripts.

Link to preprint: https://doi.org/10.26434/chemrxiv-2021-mxrxw-v2
