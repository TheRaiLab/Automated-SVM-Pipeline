##### ML Packvariabs #####
library(caret)
library(randomForest)
library(readxl)
library(reshape)
library(kernlab)
library(readr)
library(caTools)
library(iml)
library(pushoverr)
library(forcats)
library(pROC)
library(imbalance)
library(ggcorrplot)
library(ROSE)
library(ggrepel)
library(precrec)
library(glmnet)
library(e1071)
library(patchwork)
library(dplyr)
library(tibble)
library(plyr)
library(DMwR)
library(writexl)


## NOTE: The reason AUC Step increases aren't fluid are because the test
##       group only has 24 samples, due to 80:20 split


##### Directory Set #####


desktop_path <- file.path(Sys.getenv("USERPROFILE"), "Desktop")  # Windows

# Define main output directory
main_dir <- file.path(desktop_path, "Machine Learning Results")
dir.create(main_dir, recursive = TRUE)
dir.create(paste0(main_dir,"/Univariate SVM ROC-Curves"))

set.seed(1001)
##### Test and Train Splits #####
set.seed(1001)

# Update "PATH" to a path location to your input file. Ensure your input file
# has the first column as your categorical (two class only) predictor variable
# Ensure all features in your data are numeric, and contain no missing values.



SVMData<-read.csv("PATH") # For comma separated
SVMData<-read_xlsx("PATH")# For Excel format
ClinicalID<-SVMData[c(1)]

Proteomics<-SVMData[c(-1)]

SVMData<-data.frame(ClinicalID,Proteomics)
SVMData<-SVMData


ClinicalID<-SVMData[c(1)]

names(ClinicalID)<-"variab"

names(ClinicalID)<-"variab"

SVMData<-SVMData[c(-1)]
SVMData<-cbind(ClinicalID,SVMData)

SVMData$variab<- factor(SVMData$variab, labels = c("No","Yes"))
SVMData<-SVMData[c(1:10)]

splitR<-sample.split(SVMData$variab, SplitRatio = 0.70)
trainData<-as.data.frame(subset(SVMData, splitR == TRUE))
testData<-as.data.frame(subset(SVMData, splitR == FALSE))


trainData <- SMOTE(variab ~ ., data = SVMData, perc_over = 100)

##### Fully Automated Individual SVM Analysis #####

set.seed(1001)
ModelList<- list()

fControl <- trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 3,
                         classProbs = TRUE,
                         search = "random",
                         verboseIter = FALSE,
                         summaryFunction = twoClassSummary)
feat<-trainData[c(-1)]


pb <- txtProgressBar(min = 0, max = length(feat), style = 3)
for (i in 1:length(feat)) {
  feature <- colnames(feat)[i]
  formula1 <- as.formula(paste("variab ~", feature))
  
  # suppress all output (messvariabs, warnings, prints)
  capture.output({
    SVMModels <- suppressWarnings(
      train(formula1, data = trainData,
            method = "svmLinear",
            metric = "ROC",
            tuneLength = 10,
            preProcess = c("center", "scale"),
            trControl = fControl)
    )
  })
  
  ModelList[[feature]] <- SVMModels
  #print(paste0("Currently on Model Iteration  ", i))
  setTxtProgressBar(pb, i)
}

close(pb) 


PredsDF <- data.frame(variab=factor(rep(1, nrow(testData))))
ProbsDF <- data.frame(variab=factor(rep(1, nrow(testData))))

# Predictions of Best Models
for (i in 1:length(ModelList)){
  
  predictionsRAW<-predict(ModelList[i], testData, type="raw")
  ppp<-as.data.frame(predictionsRAW[1])
  PredsDF<-cbind(PredsDF,ppp)
  predictionsPROB<-as.data.frame(predict(ModelList[i], testData, type="prob"))
  ProbsDF<-cbind(ProbsDF,predictionsPROB[2])

  
}

#Remove outcome variable being treated as a feature name
names(ProbsDF)<-names(SVMData)
names(PredsDF)<-names(SVMData)

PredsDF<-PredsDF[c(-1)]
ProbsDF<-ProbsDF[c(-1)]

AccuracyLists<- list()
# Lists within a List of Accuracy Statistics (Nightmare getting out)

for (i in 1:length(PredsDF)){
  
  ConfMatrices<- confusionMatrix(testData$variab, PredsDF[[i]])
  AccuracyLists[[i]]<-ConfMatrices
}

# Create a data frame to store the accuracy statistics
TotalAccuracysSVM <- data.frame(feature = character(),
                                Accuracy = numeric(),
                                pValue = numeric(),
                                Sensitivity = numeric(),
                                Specificity = numeric(),
                                PositivePred = numeric(),
                                NegativePred = numeric(),
                                Precision = numeric(),
                                Recall = numeric(),
                                stringsAsFactors = FALSE)

for (i in 1:length(AccuracyLists)) {
  feature <- colnames(PredsDF)[[i]]
  cm <- AccuracyLists[[i]]
  # Single iterations through each confusion matrix within the list
  # each iter takes the stats below, chomp and change the different
  # stats from byClass for accuracies, which is a component of the cm structure
  accuracy <- cm$overall[1]
  pval<- cm$overall[6]
  sensitivity <- cm$byClass[1]
  specificity <- cm$byClass[2]
  PosPred<- cm$byClass[3]
  NegPred<- cm$byClass[4]
  Precise<- cm$byClass[5]
  Recalls<- cm$byClass[6]
  
  # Store the final accuracy statistics in the data frame
  TotalAccuracysSVM<- rbind(TotalAccuracysSVM,
                            data.frame(feature,
                                       Accuracy = accuracy,
                                       pVal = pval,
                                       Sensitivity = sensitivity,
                                       Specificity = specificity,
                                       PositivePred = PosPred,
                                       NegativePred = NegPred,
                                       Precision = Precise,
                                       Recall = Recalls,
                                       stringsAsFactors = FALSE,
                                       row.names = NULL))
}


write.csv(TotalAccuracysSVM,paste0(main_dir,"/Univariate SVM Statistics.csv"), row.names=FALSE)


main_dir

##### Automated ROC-Curve Lists #####

ROCListsSVM<-list()
x<-0
for (i in 1:length(ProbsDF)){ ### For SVM use ProbsDF instead of PredsDF
  features<-colnames(ProbsDF)[i]
  y<-ProbsDF[[i]]
  x<-roc(as.numeric(testData$variab), as.numeric(y), plot=T,
         auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
         print.auc=TRUE, ci=TRUE,
         legacy.axes = TRUE)
  ROCListsSVM[[features]]<-x
}

##### Uni-variate SVM ROC Curves #####

for (i in 1:length(ROCListsSVM)){
  Iter<-ROCListsSVM[[i]]
  dats<-ci.se(Iter, conf.level=0.95, method=c("bootstrap"),
              boot.n = 5000,
              boot.stratified = TRUE,
              reuse.auc=TRUE,
              specificities=seq(0, 1, l=25),
              plot=FALSE)
  
  dat.ci <- data.frame(x = as.numeric(rownames(dats)),
                       lower = dats[, 1],
                       upper = dats[, 3])
  
  SVMROC<-ggroc(Iter, legacy.axes = T) +
    theme_minimal() +
    geom_abline(slope=1, intercept = 1, linetype = "dashed", alpha=0.7, color = "grey") +
    coord_equal() +
    geom_ribbon(data = dat.ci, aes(x = 1-x, ymin = lower, ymax = upper), fill = "steelblue", alpha= 0.2)+
    xlab(expression("1-Specificity"))+
    ylab(expression("Sensitivity"))+
    ggtitle(paste("SVM ROC Curves:",names(PredsDF[i])),paste0("AUC:",round(Iter$auc,3)))+
    theme_bw(base_size = 16)

  plot(SVMROC)
  
  ggsave(
    file= paste0(names(PredsDF)[i],".tiff"),
    plot = last_plot(),
    device = NULL,
    path = paste0(main_dir,"/Univariate SVM ROC-Curves"),
    scale = 1,
    width = 1700,
    height = 1700,
    units = c("px"),
    dpi = 350,
    limitsize = TRUE,
    bg = NULL,
  )
}






warnings()

##### SVM WIP Feature Selection (Linear) #####

set.seed(1001)
# Can change repeatedcv to cv and remove repeats for speed, but you'll lose robustness
train_ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE,
  returnResamp = "all"
)


all_features <- setdiff(names(trainData), "variab")
remaining_features <- all_features
selected_features <- c()
performance_log <- data.frame(Size = integer(), Feature = character(), AUC = numeric(), Lower_CI = numeric(), Upper_CI = numeric())
max_features <- length(trainData$variab)/10

if(max_features > length(trainData[c(-1)])){
   max_features<-length(trainData[c(-1)])-1
}

for (i in 1:max_features) {
  candidate_results <- data.frame(Feature = character(), AUC = numeric(), Lower_CI = numeric(), Upper_CI = numeric(), stringsAsFactors = FALSE)
  print(i)
  
  for (feature in remaining_features) {
    # Create the feature set for the current iteration
    candidate_set <- c(selected_features, feature)
    formula <- as.formula(paste("variab ~", paste(candidate_set, collapse = " + ")))
    
    # Train the SVM model
    model <- train(
      formula,
      data = trainData,
      method = "svmLinear",
      trControl = train_ctrl,
      metric = "ROC",
      preProcess = c("center", "scale"),
      tuneLength = 5
    )
    
    # Extract the AUC from the resample results
    auc_scores <- model$resample$ROC
    
    # Calculate the mean AUC
    mean_auc <- mean(auc_scores)
    
    # Calculate the 95% confidence interval
    auc_sd <- sd(auc_scores)
    n_folds <- length(auc_scores)
    ci_lower <- mean_auc - 1.96 * (auc_sd / sqrt(n_folds))
    ci_upper <- mean_auc + 1.96 * (auc_sd / sqrt(n_folds))
    
    # Store result for the candidate feature
    candidate_results <- rbind(candidate_results, data.frame(Feature = feature, AUC = mean_auc, Lower_CI = ci_lower, Upper_CI = ci_upper))
  }
  
  # Select the feature with the highest AUC
  best_candidate <- candidate_results %>% arrange(desc(AUC)) %>% slice(1)
  selected_features <- c(selected_features, best_candidate$Feature)
  remaining_features <- setdiff(remaining_features, best_candidate$Feature)
  
  # Log performance
  performance_log <- rbind(performance_log, data.frame(
    Size = length(selected_features),
    Feature = best_candidate$Feature,
    AUC = best_candidate$AUC,
    Lower_CI = best_candidate$Lower_CI,
    Upper_CI = best_candidate$Upper_CI
  ))
  
  # Update
  cat("Step", i, "- Added feature:", best_candidate$Feature, " | AUC:", best_candidate$AUC, "\n")
}

write_xlsx(performance_log,paste0(main_dir,"/Linear SVM Performance.xlsx"))



##### SVM WIP Feature Selection (Radial) #####

set.seed(1001)
# Setup for cross-validation
train_ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE,
  returnResamp = "all"
)


all_features <- setdiff(names(trainData), "variab")
remaining_features <- all_features
selected_features <- c()
performance_log <- data.frame(Size = integer(), Feature = character(), AUC = numeric(), Lower_CI = numeric(), Upper_CI = numeric())
max_features <- length(trainData$variab)/10

if(max_features > length(trainData[c(-1)])){
  max_features<-length(trainData[c(-1)])-1
}

for (i in 1:max_features) {
  candidate_results <- data.frame(Feature = character(), AUC = numeric(), Lower_CI = numeric(), Upper_CI = numeric(), stringsAsFactors = FALSE)
  print(i)
  
  for (feature in remaining_features) {

    candidate_set <- c(selected_features, feature)
    formula <- as.formula(paste("variab ~", paste(candidate_set, collapse = " + ")))
    

    model <- train(
      formula,
      data = trainData,
      method = "svmRadial",
      trControl = train_ctrl,
      metric = "ROC",
      preProcess = c("center", "scale"),
      tuneLength = 10
    )
  
    auc_scores <- model$resample$ROC

    mean_auc <- mean(auc_scores)
    
    auc_sd <- sd(auc_scores)
    n_folds <- length(auc_scores)
    ci_lower <- mean_auc - 1.96 * (auc_sd / sqrt(n_folds))
    ci_upper <- mean_auc + 1.96 * (auc_sd / sqrt(n_folds))
    

    candidate_results <- rbind(candidate_results, data.frame(Feature = feature, AUC = mean_auc, Lower_CI = ci_lower, Upper_CI = ci_upper))
  }
  

  best_candidate <- candidate_results %>% arrange(desc(AUC)) %>% slice(1)
  selected_features <- c(selected_features, best_candidate$Feature)
  remaining_features <- setdiff(remaining_features, best_candidate$Feature)
  

  performance_log <- rbind(performance_log, data.frame(
    Size = length(selected_features),
    Feature = best_candidate$Feature,
    AUC = best_candidate$AUC,
    Lower_CI = best_candidate$Lower_CI,
    Upper_CI = best_candidate$Upper_CI
  ))
  

  cat("Step", i, "- Added feature:", best_candidate$Feature, " | AUC:", best_candidate$AUC, "\n")
}

write_xlsx(performance_log,paste0(main_dir,"/Radial SVM Performance.xlsx"))


##### Visualization of Feature Selection #####

LinearResult <- read_xlsx(paste0(main_dir,"/Linear SVM Performance.xlsx"))
RadialResult <- read_xlsx(paste0(main_dir,"/Radial SVM Performance.xlsx"))


# Function to process a result dataframe
process_result <- function(df, model_name) {
  Feats <- df$Feature  # Change index if needed
  feat_vector <- unlist(strsplit(Feats, ","))
  feat_vector <- trimws(feat_vector)
  feat_df <- data.frame(Feature = feat_vector)
  PlotDf <- cbind(df[1], feat_df, df[3], df[4], df[5])
  colnames(PlotDf) <- c("Number", "Protein", "ROC", "Lower","Upper")
  PlotDf <- PlotDf[1:max_features, ]
  PlotDf$Model <- model_name
  return(PlotDf)
}

# Process each model
RadialPlotDf <- process_result(RadialResult, "Radial")
LinearPlotDf <- process_result(LinearResult, "Linear")

# Combine all models
AllPlotDf <- bind_rows(RadialPlotDf, LinearPlotDf)

AllPlotDf<-arrange(AllPlotDf,desc(AllPlotDf$ROC))
Opt<-AllPlotDf[1,1]

ROC_SVM <- ggplot(AllPlotDf, aes(x = Number, y = ROC, color = Model)) +
  geom_vline(xintercept = Opt, linetype="dashed",colour="grey", linewidth=1)+
  geom_ribbon(aes(ymin=Lower, ymax=Upper), linetype=2, alpha=0.1)+
  geom_point(size = 1.5, alpha = 1, na.rm = TRUE) +
  geom_line(aes(group = Model)) +
  scale_color_manual(values = c("Radial" = "#0173b2", 
                                "Linear" = "#de8f05")) +
  xlab("Number of Features") +
  ylab("ROC-AUC Score (10-Fold Cross Validated)") +
  ggtitle("Forward Feature Selection SVM", 
          subtitle = "(95% CI's from 10-Fold Cross Validation, n=3 repeats)") +
  theme_bw(base_size = 18) +
  theme(legend.key.size = unit(1.5, "lines")) +  # Increase legend key size
  guides(color = guide_legend(override.aes = list(size = 4)))+
  geom_text_repel(aes(label = Protein), colour = "black", size = 4, nudge_y = 0.001, segment.color = NA, show.legend = FALSE)
# Increase legend point size

# Show plot
ROC_SVM

ggsave(
  'All Comparisons.tiff',
  plot = ggplot2::last_plot(),
  device = NULL, 
  path = main_dir,
  scale = 1,
  width = 4500, #35
  height = 3000, #30
  units = c("px"),
  dpi = 350,
  limitsize = FALSE,
  bg = NULL,
)



##### Finalized Model Building #####
set.seed(1001)
splitR<-sample.split(SVMData$variab, SplitRatio = 0.70)
trainData<-as.data.frame(subset(SVMData, splitR == TRUE))
testData<-as.data.frame(subset(SVMData, splitR == FALSE))


trainData <- SMOTE(variab ~ ., data = SVMData, perc_over = 100)


LinearPlotDf<-arrange(LinearPlotDf,desc(LinearPlotDf$ROC))
RadialPlotDf<-arrange(RadialPlotDf,desc(RadialPlotDf$ROC))

HighLIN<-LinearPlotDf[1,1]
HighRAD<-RadialPlotDf[1,1]

if(HighLIN > HighRAD){
  BestModel <- LinearPlotDf
} else {
  BestModel <- RadialPlotDf
}


if(HighLIN > HighRAD){
  method <- "svmLinear"
} else {
  method <- "svmRadial"
}

OptimalFeats<-BestModel$Protein[1:Opt]

trainID<-trainData[c(1)]
testID<-testData[c(1)]

trainData<-trainData[,colnames(trainData) %in% OptimalFeats]
testData<-testData[,colnames(testData) %in% OptimalFeats]

trainData<-cbind(trainID,trainData)
testData<-cbind(testID,testData)

set.seed(1001)
# Setup for cross-validation
train_ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  verboseIter = F,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE,
  returnResamp = "all"
)


# Train the SVM model
model <- train(variab~.,
               data = trainData,
               method = method,
               trControl = train_ctrl,
               metric = "ROC",
               preProcess = c("center", "scale"),
               tuneLength = 5
)

Predictions<-predict(model,testData)

write_rds(model,paste0(main_dir,"/Optimised Final Model.rds"))


##### Decision Boundary Projection #####

# Function to create a 2D grid across PC1 and PC2
create_grid <- function(data, resolution = 150) {
  ranges <- apply(data, 2, range)
  margin <- 0.1 * (ranges[2,] - ranges[1,])
  x_range <- seq(ranges[1, 1] - margin[1], ranges[2, 1] + margin[1], length.out = resolution)
  y_range <- seq(ranges[1, 2] - margin[2], ranges[2, 2] + margin[2], length.out = resolution)
  expand.grid(PC1 = x_range, PC2 = y_range)
}

# Assuming `trainData`, `SVMModels`, and `best_models` already exist
FINALMODEL <- model

# Prepare feature matrix and target
X <- trainData[, -1]         # All 17 features
y <- trainData$variab           # Target variable (factor)

# Perform PCA on training features
pca_result <- prcomp(X, center = TRUE, scale. = TRUE)
summary_pca <- summary(pca_result)

# Project training data onto first two PCs
pca_scores <- predict(pca_result, X)
pca_data <- as.data.frame(pca_scores[, 1:2])
pca_data$class <- y

# Step 1: Create grid in PC1-PC2 space
grid <- create_grid(pca_data[, 1:2])


# Step 2: Construct full-length PC space (using all PCs, not just PC1/PC2)
n_pcs <- ncol(pca_result$x)
grid_all_pc <- matrix(0, nrow = nrow(grid), ncol = n_pcs)
grid_all_pc[, 1:2] <- as.matrix(grid)  # Fill PC1 and PC2; leave others at 0 (mean)

# Step 3: Inverse transform to original feature space (17D)
grid_orig_space <- grid_all_pc %*% t(pca_result$rotation)
for (i in 1:ncol(grid_orig_space)) {
  grid_orig_space[, i] <- grid_orig_space[, i] * pca_result$scale[i] + pca_result$center[i]
}
grid_df <- as.data.frame(grid_orig_space)
colnames(grid_df) <- colnames(X)

# Step 4: Predict using full model
grid$prediction_full <- predict(FINALMODEL, grid_df)

# Step 5: Plot decision boundary using PC1/PC2 projection
p1 <- ggplot() +
  geom_tile(data = grid, aes(x = PC1, y = PC2, fill = prediction_full), alpha = 0.3) +
  geom_point(data = pca_data, aes(x = PC1, y = PC2, color = class), size = 3) +
  scale_fill_manual(values = c("No" = "skyblue", "Yes" = "salmon"), name = "Prediction") +
  scale_color_manual(values = c("No" = "blue", "Yes" = "red"), name = "Actual") +
  labs(
    title = "Full SVM Model (8D) Decision Boundary in PCA Space",
    subtitle = paste("Cross-validation AUC:", round(mean(model$resample$ROC), 4)),
    x = paste0("PC1 (", round(summary_pca$importance[2, 1] * 100, 1), "% variance)"),
    y = paste0("PC2 (", round(summary_pca$importance[2, 2] * 100, 1), "% variance)")
  ) +
  theme_bw(base_size = 16)

p1

ggsave(
  'Decision Boundary.tiff',
  plot = ggplot2::last_plot(),
  device = NULL, 
  path = main_dir,
  scale = 1,
  width = 3500, #35
  height = 3000, #30
  units = c("px"),
  dpi = 350,
  limitsize = FALSE,
  bg = NULL,
)

##### SHapley Additive exPlanations #####
SVMModels<-model

# Create predictor wrapper
predictor <- Predictor$new(SVMModels, data = trainData[, -1])





# Default empty list for storvariab
shap_values_list <- list()

# Loop over all observations within trainData to compute Shapley values
for (i in 1:nrow(trainData)) {
  shapley <- Shapley$new(predictor, x.interest = trainData[i, -1])
  
  # Filter for class "Yes" and select feature + SHAP vals
  shap_row <- shapley$results %>%
    filter(class == "Yes") %>%
    select(feature, phi)
  
  # Transpose SHAP values into standard layout
  shap_row_t <- setNames(as.data.frame(t(shap_row$phi)), shap_row$feature)
  
  # Store in list
  shap_values_list[[i]] <- shap_row_t
  print(i)
}

# Combine all rows into one dataframe (features as columns)
SHAPDF <- do.call(rbind, shap_values_list)

# Extract feature values from trainData (make sure row count matches)
FeatureVals <- trainData[, -1]
FeatureVals <- FeatureVals[1:nrow(SHAPDF), ]

# Scale feature values, is important for visualisation
FeatureVals_scaled <- scale(FeatureVals)

# Add observation ID to both SHAP and Feature dataframes
SHAPDF$obs_id <- 1:nrow(SHAPDF)
FeatureVals_scaled <- as.data.frame(FeatureVals_scaled)
FeatureVals_scaled$obs_id <- 1:nrow(FeatureVals_scaled)

# Melt both dataframes to long format, keeping obs_id for alignment
SHAP_long <- melt(SHAPDF, id.vars = "obs_id", variable.name = "Feature", value.name = "SHAP_Value")
colnames(SHAP_long)<-c("obs_id","Feature","SHAP_Value") # Issues with melt not assigning correct variable name, temp fix for it 

FeatureVals_long <- melt(FeatureVals_scaled, id.vars = "obs_id", variable.name = "Feature", value.name = "Feature Value")
colnames(FeatureVals_long)<-c("obs_id","Feature","Feature Value") # Issues with melt not assigning correct variable name, temp fix for it 

# Merge by obs_id and Feature to ensure correct alignment
combined_long <- merge(SHAP_long, FeatureVals_long, by = c("obs_id", "Feature"))



# Reorder factor levels of Feature by importance
#combined_long$Feature <- factor(combined_long$Feature, levels = feature_order)

# Plot SHAP summary
SHAP_plot <- ggplot(combined_long, aes(x = SHAP_Value, y = Feature, color = `Feature Value`)) +
  geom_jitter(width = 0.01, height = 0.2, alpha = 1) +
  scale_color_gradient2(low = "dodgerblue3", mid = "gray90", high = "firebrick3", midpoint = 0) +
  theme_bw(base_size = 16) +
  labs(
    title = "SHAP Summary Plot",
    subtitle = "(SHAP values calculated for all observations in train cohort)",
    x = "SHAP Value",
    y = ""
  ) +
  theme(axis.text.y = element_text(size = 16, family = "Arial"))+
  scale_y_discrete(limits = names(sort(tapply(abs(SHAP_long$SHAP_Value), SHAP_long$Feature, mean), decreasing = F)))

SHAP_plot

ggsave(
  'SHapley Additive exPlanations.tiff',
  plot = ggplot2::last_plot(),
  device = NULL, 
  path = main_dir,
  scale = 1,
  width = 3500, #35
  height = 3000, #30
  units = c("px"),
  dpi = 350,
  limitsize = FALSE,
  bg = NULL,
)

##### CM MATRIX #####

pre2<-predict(SVMModels,testData)



cm <- confusionMatrix(pre2, testData$variab)
desired_order <- c("No", "Yes")
# Create table for plot
cm_table <- table(Predicted = pre2, Actual = testData$variab)
cm_df <- as.data.frame(cm_table)

# Reapply levels for ggplot control
cm_df$Predicted <- factor(cm_df$Predicted, levels = rev(desired_order))  # Flip y-axis to put Declined on top
cm_df$Actual <- factor(cm_df$Actual, levels = desired_order)

# Plot
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 8) +
  scale_fill_gradientn(colors = rev(hcl.colors(25, "OrRd"))) +
  scale_y_discrete(limits = rev(desired_order)) + 
  theme_bw(base_size = 18) +
  labs(
    title = "Confusion Matrix: Predicted vs Actual",
    x = "Actual Outcome",
    y = "Predicted Outcome"
  )

ggsave(
  'Confusion Matrix.tiff',
  plot = ggplot2::last_plot(),
  device = NULL, 
  path = main_dir,
  scale = 1,
  width = 3500, #35
  height = 3000, #30
  units = c("px"),
  dpi = 350,
  limitsize = FALSE,
  bg = NULL,
)

##### ROC #####

pre1<-predict(SVMModels,testData, type="prob")

pre2<-predict(SVMModels,testData, type="raw")

g<-confusionMatrix(pre2,testData$variab)

x<-roc(as.numeric(testData$variab), as.numeric(pre1$Yes), plot=T,
       auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
       print.auc=TRUE, ci=TRUE,
       legacy.axes = TRUE)

dats<-ci.se(x, conf.level=0.95, method=c("bootstrap"),
            boot.n = 5000,
            boot.stratified = TRUE,
            reuse.auc=TRUE,
            specificities=seq(0, 1, l=100),
            plot=FALSE)

dat.ci <- data.frame(x = as.numeric(rownames(dats)),
                     lower = dats[, 1],
                     upper = dats[, 3])

SVMROC<-ggroc(x, legacy.axes = T) +
  theme_minimal() +
  geom_abline(slope=1, intercept = 1, linetype = "dashed", alpha=0.7, color = "grey") +
  coord_equal() +
  geom_ribbon(data = dat.ci, aes(x = 1-x, ymin = lower, ymax = upper), fill = "steelblue", alpha= 0.2)+
  xlab(expression("1-Specificity"))+
  ylab(expression("Sensitivity"))+
  ggtitle("SVM ROC Curve Optimal Panel",paste0("AUC:",round(x$auc,3)))+
  theme_bw(base_size = 16)

SVMROC

ggsave(
  'Optimal Model ROC Curve.tiff',
  plot = ggplot2::last_plot(),
  device = NULL, 
  path = main_dir,
  scale = 1,
  width = 2000, #35
  height = 2000, #30
  units = c("px"),
  dpi = 350,
  limitsize = FALSE,
  bg = NULL,
)



















