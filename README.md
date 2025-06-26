# SVM Machine Learning Pipeline

A comprehensive R-based machine learning pipeline for binary classification using Support Vector Machines (SVM) with automated feature selection, model comparison, and interpretability analysis.

## Overview

This pipeline performs end-to-end machine learning analysis including:
- Automated univariate SVM analysis
- Forward feature selection with cross-validation
- Model comparison (Linear vs Radial SVM)
- SHAP-based model interpretability
- Comprehensive visualization and reporting

## Features

### **Automated Analysis**
- Individual feature analysis with progress tracking
- Automated ROC curve generation for each feature
- Statistical performance metrics calculation

### **Feature Selection** 
- Forward stepwise feature selection
- Cross-validated performance evaluation
- Confidence interval estimation for robust selection

### **Model Comparison**
- Linear SVM vs Radial SVM comparison
- Automated optimal model selection
- Performance visualization with confidence bands

### **Model Interpretability**
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance visualization
- Decision boundary projection in PCA space

### **Comprehensive Reporting**
- ROC curves with bootstrap confidence intervals
- Confusion matrices
- Performance comparison plots
- Publication-ready TIFF outputs

## Requirements

### Required R Packages
```r
library(caret)           # Machine learning framework
library(randomForest)    # Random forest algorithms
library(readxl)          # Excel file reading
library(reshape)         # Data reshaping
library(kernlab)         # SVM kernels
library(readr)           # Fast data reading
library(caTools)         # Data splitting
library(iml)             # Interpretable ML
library(pushoverr)       # Notifications
library(forcats)         # Factor handling
library(pROC)            # ROC analysis
library(imbalance)       # Imbalanced data handling
library(ggcorrplot)      # Correlation plots
library(ROSE)            # Sampling techniques
library(ggrepel)         # Text repelling in plots
library(precrec)         # Precision-recall curves
library(glmnet)          # Regularized regression
library(e1071)           # SVM implementation
library(patchwork)       # Plot composition
library(dplyr)           # Data manipulation
library(tibble)          # Modern data frames
library(plyr)            # Data manipulation
library(DMwR)            # Data mining with R
library(writexl)         # Excel writing
```

## Usage

### 1. Data Preparation
```r
# Update file path to your data
SVMData <- read.csv("PATH")      # For CSV files
SVMData <- read_xlsx("PATH")     # For Excel files
```

**Data Requirements:**
- First column: Binary categorical outcome variable (will be converted to "No"/"Yes")
- Remaining columns: Numeric predictor variables
- No missing values
- All features should be numeric

### 2. Configuration
```r
# Set random seed for reproducibility
set.seed(1001)

# Adjust data subset if needed (currently uses first 10 columns)
SVMData <- SVMData[c(1:10)]

# Modify train/test split ratio (default: 70/30)
splitR <- sample.split(SVMData$variab, SplitRatio = 0.70)
```

### 3. Run Analysis

The pipeline runs automatically through several stages:

#### Stage 1: Univariate Analysis
- Individual SVM models for each feature
- ROC curve generation
- Statistical performance metrics

#### Stage 2: Feature Selection
- Forward stepwise selection
- Cross-validated performance evaluation
- Separate analysis for Linear and Radial SVM

#### Stage 3: Model Comparison
- Performance visualization
- Optimal model selection
- Feature importance ranking

#### Stage 4: Final Model
- Training with optimal features
- SHAP analysis for interpretability
- Decision boundary visualization

## Output Files

All outputs are saved to `~/Desktop/Machine Learning Results/`:

### **Statistical Reports**
- `Univariate SVM Statistics.csv` - Individual feature performance
- `Linear SVM Performance.xlsx` - Linear SVM feature selection results
- `Radial SVM Performance.xlsx` - Radial SVM feature selection results

### **Visualizations**
- `All Comparisons.tiff` - Feature selection comparison plot
- `Decision Boundary.tiff` - Model decision boundary in PCA space
- `SHapley Additive exPlanations.tiff` - SHAP summary plot
- `Confusion Matrix.tiff` - Final model confusion matrix
- `Optimal Model ROC Curve.tiff` - Final model ROC curve
- Individual ROC curves in `/Univariate SVM ROC-Curves/` folder

### **Model Files**
- `Optimised Final Model.rds` - Trained model object for future predictions
- 
### Cross-Validation Settings
```r
train_ctrl <- trainControl(
  method = "repeatedcv",    # Repeated cross-validation
  number = 10,              # 10-fold CV
  repeats = 3,              # 3 repetitions
  classProbs = TRUE,        # Enable probability predictions
  summaryFunction = twoClassSummary  # Binary classification metrics
)
```

### Feature Selection Limits
```r
max_features <- length(trainData$variab)/10  # Maximum features to select
```

### Data Balancing
```r
trainData <- SMOTE(variab ~ ., data = SVMData, perc_over = 100)
```

## Model Details

### SVM Configurations
- **Linear SVM**: `method = "svmLinear"` with tuneLength = 5
- **Radial SVM**: `method = "svmRadial"` with tuneLength = 10
- **Preprocessing**: Center and scale normalization
- **Metric**: ROC-AUC for model selection

### Performance Evaluation
- 10-fold repeated cross-validation (3 repeats)
- Bootstrap confidence intervals (5000 iterations)
- AUC, sensitivity, specificity, precision, recall metrics

### Adjust Cross-Validation
```r
# For faster execution (less robust)
train_ctrl <- trainControl(
  method = "cv",          # Simple CV instead of repeated
  number = 5,             # Fewer folds
  repeats = 1             # Single repetition
)
```

### Modify Feature Selection
```r
# Change maximum features
max_features <- 5  # Select only top 5 features, you can subset as needed

# Alternative: Use percentage of samples
max_features <- ceiling(nrow(trainData) * 0.1)
```

## Troubleshooting
Any issues found please report and forward to mclarnon-t1@ulster.ac.uk

**Memory errors with large datasets:**
- Reduce `tuneLength` parameter
- Use fewer CV repeats
- Consider data subsampling

**Missing packages:**
```r
# Install all required packages
install.packages(c("caret", "randomForest", "readxl", "reshape", 
                   "kernlab", "readr", "caTools", "iml", "pushoverr", 
                   "forcats", "pROC", "imbalance", "ggcorrplot", 
                   "ROSE", "ggrepel", "precrec", "glmnet", "e1071", 
                   "patchwork", "dplyr", "tibble", "plyr", "DMwR", 
                   "writexl"))
```


### Computational Complexity
- **Univariate analysis**: ~1-2 minutes per feature
- **Feature selection**: ~5-10 minutes per feature combination, dependant on feature panel size
- **SHAP analysis**: ~1-2 minutes per observation

### Optimization Tips
- Use `method = "cv"` instead of `"repeatedcv"` for faster execution
- Reduce `tuneLength` for quicker hyperparameter search
- Consider parallel processing with `doParallel` package

## Citation

If you use this pipeline in your research, please cite the paper that utilized this pipeline first;
[INSERT DOI ONCE PAPER IS PUBLISHED]

## License

This code is provided as-is for research and educational purposes. Please ensure you have appropriate licenses for all required R packages.

---

**Note**: Update the `PATH` variable with your actual data file location before running the analysis.
