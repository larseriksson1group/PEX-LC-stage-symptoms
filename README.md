# Predicting lung cancer stage at diagnosis based on self-reported symptoms and background factors using machine learning models

## Introduction
This study characterised and compared background factors and symptoms at diagnosis across patients with non-advanced lung cancer, advanced lung cancer, and cancer-free controls. Machine learning models (regularized logistic regression, random forest, and extreme gradient boosting) were trained and evaluated to identify variables that contribute to the detection of both early- and late-stage lung cancer, and to assess the potential predictive value of detailed patient-reported symptoms data beyond background risk factors. 

The results of the study are published in _:

Gustavell, T., Sissala, N., Babačić, H. et al. Predicting lung cancer stage at diagnosis based on self-reported symptoms and background factors using machine learning models. [Journal], [Year]. [DOI]

## Contents
This repository contains all code used for the machine learning analysis performed in the study, including result interpretation and visualization. **Note:** The data are not publicly available to protect study participants' privacy. Code and results are provided for transparency and reproducibility of the analytical approach.

## Repository structure

```text
├── code
	└── ML_analysis.R 			# Main analysis script
	├── ML_functions.R 			# Functions for model training and evaluation
	├── plotting_functions.R 	# ggplot2 themes and plotting functions
└── results
	└── ML_analysis/ 			# Output directory. Contains plots and sub-folders with output of each machine learning run.
```
## Analysis overview
`ML_analysis.R` runs the full analysis pipeline including:
- Model training — Repeated 10-fold cross-validation (CV) for regularized logistic regression, random forest, and extreme gradient boosting, separately for advanced and non-advanced lung cancer versus controls, and for three predictor sets (all variables, background variables only, symptom variables only).
- Performance evaluation — ROC-AUC, sensitivity, specificity, PPV, NPV, and balanced accuracy at the default (0.5) and Youden-optimal probability thresholds. Mean ROC curves calculated across CV folds.
- Calibration — Model calibration curves comparing predicted and true lung cancer probabilities, estimated using CalibrationCurves::val.prob.ci.2().
- Decision curve analysis — Standardised net benefit across a range of threshold probabilities using rmda.
- Variable importance — Permutation-based and model-based importance of background and symptom predictors across CV folds.

## Dependencies
The R packages used in the project are recorded in `results/ML_analysis/session_info.txt` for the analysis and visualization of machine learning results, and for each individual machine learning run in the corresponding result subfolder (e.g. `results/ML_analysis/advanced`) in `ML_session_info.txt`.

## Citation
If you use this code in your research, please cite the original publication and the code:

Article: Gustavell, T., Sissala, N., Babačić, H. et al. Predicting lung cancer stage at diagnosis based on self-reported symptoms and background factors using machine learning models. [Journal], [Year]. [DOI]

Code: Sissala, N. PEX-LC-stage-symptoms. Zenodo [doi] [Year].

## License
This project is licensed under the MIT License.
