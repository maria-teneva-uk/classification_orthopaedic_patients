library(tidyverse)
library(reshape2)
library(factoextra)
library(cluster)
library(viridis)
library(knitr)
library(kableExtra)
library(caret)
library(pROC)
library(themis) 
library(recipes)
library(patchwork)
# Import dataset in RStudio
# Rename
data<- column_2C 
# Change column names
colnames(data) <- c("Pelvic_Incidence", "Pelvic_Tilt", "Lumbar_Lordosis_Angle", "Sacral_Slope", "Pelvic_Radius", "Grade_of_Spondylolisthesis", "Class_Label")
# Summary stats
summary(data)
#check for missing values
sum(is.na(data))
# Check class distribution
data %>% count(Class_Label)
# EDA
df_visual <- data[, -which(names(data) == 'Class_Label')]
df_long <- pivot_longer(df_visual, cols = everything(), names_to = "Variable", values_to = "Value")
df_long$Variable <- gsub("_", " ", df_long$Variable)

# Boxplots
ggplot(df_long, aes(x = Variable, y = Value, fill = Variable)) +
  geom_boxplot(show.legend = FALSE) + # Hide legend for clarity
  scale_fill_viridis_d() + # Apply a pleasing color palette
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), # Rotate x labels for readability
        plot.title = element_text(hjust = 0.5)) + # Center the title
  labs(title = "Boxplots of Variables", x = "Variable", y = "Value")

# Correlation matrix
cor_data <- data
cor_data$Classifier <- ifelse(data$Class_Label == "AB", 1, 0)
cor_data <- cor_data[, -7]
cor_matrix <- cor(cor_data[,1:7])


# Convert the correlation matrix into a longer format
melted_cor_matrix <- melt(cor_matrix)
melted_cor_matrix$Var1 <- gsub("_", " ", melted_cor_matrix$Var1)
melted_cor_matrix$Var2 <- gsub("_", " ", melted_cor_matrix$Var2)

ggplot(data = melted_cor_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", value)), color = "white", size = 3) +  
  scale_fill_viridis(option = "M", direction = -1, begin = 0, end = 1, 
                     limits = c(-1, 1), name = "Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) +
  coord_fixed()

# By class KDEs
data_long <- reshape2::melt(data, id.vars = "Class_Label")

custom_colors <- c("AB" = "#238A8DFF", "NO" = "#FDE725FF")
# Plotting KDEs for all variables
ggplot(data_long, aes(x = value, fill = Class_Label)) +
  geom_density(alpha = 0.7) +
  facet_wrap(~ variable, scales = "free") +
  scale_fill_manual(values=custom_colors) +
  theme_minimal() +
  labs(title = "KDE of Variables by Class",
       x = "Value",
       y = "Density") +
  theme(legend.title = element_blank())

# Unsupervised learning preprocessing
data_kmeans <- data[, -which(names(data) == "Class_Label")]
df_transformed <- data_kmeans
# Log transformation for each column except the last one
for(i in 1:(ncol(data_kmeans) - 1)){
  shift_value <- abs(min(data_kmeans[,i], na.rm = TRUE)) + 1
  df_transformed[,i] <- log(data_kmeans[,i] + shift_value)
}

# Standardisation and PCA
df_standardized <- scale(df_transformed)
sum(is.na(df_standardized))
pca_result <- prcomp(df_standardized)
df_pca <- as.data.frame(pca_result$x)

# PCA variance explained by each component
var_explained <- summary(pca_result)$importance[2,] * 100
print(var_explained)
cum_var_explained <- cumsum(var_explained)
print(cum_var_explained)

# Elbow Plot
set.seed(91) # for reproducibility
fviz_nbclust(df_pca, kmeans, method = "wss") +
  labs(subtitle = "Elbow Method") +
  theme_minimal()

#K-Means 2 clusters calculate centroids in PCA space and plot classification
kmeans_result <- kmeans(df_pca, centers = 2, nstart = 310)
df_pca$cluster <- factor(kmeans_result$cluster)
centroids <- aggregate(. ~ cluster, data=df_pca, FUN=mean)

# Plot classification for 2 clusters
plot2 <- ggplot(df_pca, aes(x = PC1, y = PC2, group = cluster)) + 
  geom_point(aes(fill = cluster), color = "black", shape = 21, size = 2, stroke = 0.5, alpha = 0.5) +  
  geom_point(data = centroids, aes(x = PC1, y = PC2, fill = cluster), shape = 23, size = 6, color = "black") +
  scale_fill_viridis(discrete = TRUE, option = "D") + 
  theme_minimal() +
  labs(title = "Cluster Plot of PCA Results with Centroids")

# Perform k-means clustering on the PCA-reduced dataset with 3 clusters
kmeans_result3 <- kmeans(df_pca, centers = 3, nstart = 310)
df_pca$cluster3 <- factor(kmeans_result3$cluster)
centroids3 <- aggregate(. ~ cluster3, data=df_pca, FUN=mean)

# Plot classification for 3 clusters
plot3 <- ggplot(df_pca, aes(x = PC1, y = PC2, group = cluster3)) + 
  geom_point(aes(fill = cluster3), color = "black", shape = 21, size = 2, stroke = 0.5, alpha = 0.5) + 
  geom_point(data = centroids3, aes(x = PC1, y = PC2, fill = cluster3), shape = 23, size = 6, color = "black") +
  scale_fill_viridis(discrete = TRUE, option = "D") +  
  theme_minimal() +
  labs(title = "Cluster Plot of PCA Results with Centroids")

#Supervised Learning
set.seed(91)
data$Class_Label <- factor(data$Class_Label, levels = c("AB", "NO"))
features <- data[, -which(names(data) == "Class_Label")]
target <- data$Class_Label
# Train-test split
set.seed(91)
trainIndex <- createDataPartition(target, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Cross-validation setup with SMOTE
set.seed(91)
control <- trainControl(method="cv", 
                        number=10, 
                        savePredictions="all", 
                        classProbs=TRUE,
                        summaryFunction=twoClassSummary, 
                        returnResamp="all", 
                        search = "grid",
                        sampling = "smote")

# Hyperparameter search grid
set.seed(91)
tuneGrid <- expand.grid(.mtry =  c(2, round(sqrt(ncol(trainData)))),  # Adjusted for the number of columns in trainData
                        .splitrule = "gini",
                        .min.node.size = c(10, 15, 17, 20, 25, 30))

# Train model with cross-validation and hyperparameter tuning
set.seed(91)  # Setting seed again for model training
rf_model <- train(Class_Label ~ ., data=trainData, method="ranger", 
                  trControl=control, metric="ROC", tuneGrid=tuneGrid, 
                  importance='impurity')

# Predict on test data
set.seed(91)
test_pred <- predict(rf_model, testData)
test_pred_prob <- predict(rf_model, testData, type="prob")

# Confusion Matrix,'AB' is the positive class
set.seed(91)
conf_matrix <- confusionMatrix(test_pred, testData$Class_Label, positive="AB")
print(conf_matrix)

# Feature Importance
set.seed(91)
importance <- varImp(rf_model, scale=FALSE)
print(importance)
plot(importance)

# Best hyperparameters
set.seed(91)
print(rf_model$bestTune)

# Tuning results
set.seed(91)
print(rf_model$results)

# Final model details
set.seed(91)
print(rf_model$finalModel)

#Plot Feature Importance in a more visually appealing way
importance_data <- data.frame(
  Feature = c("Grade of Spondylolisthesis", "Pelvic Radius", "Pelvic Incidence",
              "Pelvic Tilt", "Sacral Slope", "Lumbar Lordosis Angle"),
  Importance = c(58.17, 20.40, 19.20, 15.90, 11.85, 11.01)
)

ggplot(importance_data, aes(x = reorder(Feature, Importance), y = Importance, fill = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +  
  scale_fill_viridis(option = "D") +  
  theme_minimal() +
  labs(title = "Feature Importance",
       x = "Feature",
       y = "Importance") +
  theme(legend.title = element_blank()) 


