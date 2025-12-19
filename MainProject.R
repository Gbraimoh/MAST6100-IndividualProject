################################################################################
# MAST6100 Final Project – EDA for Diabetic Hospital Readmission Dataset
################################################################################

# Load required packages -------------------------------------------------------

library(MASS)
library(tidyverse) 
library(naniar)
library(e1071)
library(corrplot)
library(caret)
library(MASS)           # LDA
library(class)          # KNN
library(rpart)          # Decision Trees
library(rpart.plot)
library(randomForest)   # Random Forest
library(nnet)
library(pROC)


# Load dataset -----------------------------------------------------------------
df <- read.csv("diabetic_data.csv", stringsAsFactors = FALSE)

# Inspect structure ------------------------------------------------------------
str(df)                                 # view variable types
summary(df)                              # summary statistics
names(df)

################################################################################
# Handle missing values ("?" entries) ------------------------------------------
################################################################################

# Replace "?" with actual NA
df[df == "?"] <- NA

# Check missing values
colSums(is.na(df))                       # count missing per column
sum(is.na(df))                           # total missing values

################################################################################
# Remove duplicates -------------------------------------------------------------
################################################################################

sum(duplicated(df))                      # number of duplicates
df <- df[!duplicated(df), ]              # remove duplicates
sum(duplicated(df))                      # confirm removal

################################################################################
# Convert categorical variables to factors -------------------------------------
################################################################################

# Identify character columns
char_cols <- sapply(df, is.character)

# Convert all character columns to factors
df[char_cols] <- lapply(df[char_cols], factor)

str(df)                            

################################################################################
# Define response variable ------------------------------------------------------
################################################################################

# Common choice: Readmission within 30 days (binary classification)
df$readmit_binary <- ifelse(df$readmitted == "<30", "YES", "NO")
df$readmit_binary <- factor(df$readmit_binary)

# Check class balance
table(df$readmit_binary)
prop.table(table(df$readmit_binary))

################################################################################
# Numeric summaries --------------------------------------------------------------
################################################################################

# Extract numeric variables
numeric_vars <- df %>% select(where(is.numeric))

summary(numeric_vars)

################################################################################
# Histograms for numeric variables ----------------------------------------------
################################################################################

numeric_vars %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "lightgray", colour = "black") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Histograms of Numeric Variables")


################################################################################
# Density plots for numeric variables -------------------------------------------
################################################################################

numeric_vars %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  filter(!is.na(value)) %>%              # density cannot handle NA values
  ggplot(aes(value)) +
  geom_density(fill = "lightgray") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Density Plots of Numeric Variables")

################################################################################
# Boxplots for numeric variables ------------------------------------------------
################################################################################

numeric_vars %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(y = value, x = variable)) +
  geom_boxplot(fill = "lightgray") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplots of Numeric Variables")

################################################################################
# Barplots for key categorical variables ----------------------------------------
################################################################################

barplot(table(df$race),
        main = "Race Distribution",
        col = "lightgray")

barplot(table(df$gender),
        main = "Gender Distribution",
        col = "lightgray")

barplot(table(df$readmit_binary),
        main = "Readmission Within 30 Days",
        col = "lightgray")

################################################################################
# Numeric variables vs response -------------------------------------------------
################################################################################

boxplot(df$time_in_hospital ~ df$readmit_binary,
        main = "Time in Hospital by Readmission Status",
        xlab = "Readmitted Within 30 Days",
        ylab = "Time in Hospital",
        col = "lightgray")

boxplot(df$num_lab_procedures ~ df$readmit_binary,
        main = "Lab Procedures by Readmission Status",
        xlab = "Readmitted Within 30 Days",
        ylab = "Number of Lab Procedures",
        col = "lightgray")

################################################################################
# Skewness checks ---------------------------------------------------------------
################################################################################

sapply(numeric_vars, skewness)

################################################################################
# Correlation matrix (numeric variables only) -----------------------------------
################################################################################

cor_matrix <- cor(numeric_vars, use = "pairwise.complete.obs")

corrplot(cor_matrix,
         method = "color",
         tl.cex = 0.6,
         main = "Correlation Matrix of Numeric Variables")

################################################################################
# END OF EDA SECTION
################################################################################



################################################################################
# PREPARATION FOR MODELLING
################################################################################


# Start from original df (already has "?" -> NA)
df2 <- df

# Create binary outcome: readmission within 30 days
df2$readmitted30 <- ifelse(df2$readmitted == "<30", "YES", "NO")
df2$readmitted30 <- factor(df2$readmitted30, levels = c("NO", "YES"))

# Build a smaller, clean modelling dataset
df_model_small <- df2 %>%
  mutate(
    race                    = factor(race),
    gender                  = factor(gender),
    age                     = factor(age),
    admission_type_id       = factor(admission_type_id),
    discharge_disposition_id= factor(discharge_disposition_id),
    admission_source_id     = factor(admission_source_id)
  ) %>%
  dplyr::select(
    readmitted30,
    race, gender, age,
    admission_type_id, discharge_disposition_id, admission_source_id,
    time_in_hospital, num_lab_procedures, num_medications,
    num_procedures, number_diagnoses,
    number_emergency, number_inpatient, number_outpatient
  ) %>%
  drop_na()          # drop rows that have NA in any of these

str(df_model_small)
table(df_model_small$readmitted30)

################################################################################
# TRAIN–TEST SPLIT
################################################################################

set.seed(123)
train_index <- createDataPartition(df_model_small$readmitted30, p = 0.7, list = FALSE)

train <- df_model_small[train_index, ]
test  <- df_model_small[-train_index, ]

X_train <- train %>% dplyr::select(-readmitted30)
y_train <- train$readmitted30

X_test  <- test %>% dplyr::select(-readmitted30)
y_test  <- test$readmitted30


################################################################################
# 1. LOGISTIC REGRESSION (GLM)
################################################################################

glm_model <- glm(readmitted30 ~ ., data = train, family = binomial)
summary(glm_model)

glm_prob <- predict(glm_model, test, type = "response")
glm_pred <- factor(ifelse(glm_prob > 0.5, "YES", "NO"), levels = c("NO", "YES"))

confusionMatrix(glm_pred, y_test)

glm_auc <- roc(y_test, glm_prob)
auc(glm_auc)

################################################################################
# 2. LDA
################################################################################

lda_model <- lda(readmitted30 ~ ., data = train)
lda_pred  <- predict(lda_model, X_test)$class
confusionMatrix(lda_pred, y_test)

lda_prob <- predict(lda_model, X_test)$posterior[,2]
lda_auc  <- roc(y_test, lda_prob)
auc(lda_auc)

################################################################################
# 3. KNN  (numeric predictors only for scaling)
################################################################################

num_cols <- c("time_in_hospital", "num_lab_procedures", "num_medications",
              "num_procedures", "number_diagnoses",
              "number_emergency", "number_inpatient", "number_outpatient")

X_train_num <- train[, num_cols]
X_test_num  <- test[, num_cols]

preproc <- preProcess(X_train_num, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train_num)
X_test_scaled  <- predict(preproc, X_test_num)

knn_pred <- knn(train = X_train_scaled,
                test  = X_test_scaled,
                cl    = y_train,
                k = 5)

confusionMatrix(knn_pred, y_test)

################################################################################
# 4. RANDOM FOREST
################################################################################

rf_model <- randomForest(readmitted30 ~ ., data = train, ntree = 300, mtry = 5)
rf_pred  <- predict(rf_model, test)
confusionMatrix(rf_pred, y_test)

rf_prob <- predict(rf_model, test, type = "prob")[,2]
rf_auc  <- roc(y_test, rf_prob)
auc(rf_auc)

################################################################################
# 5. DEEP LEARNING (using numeric predictors only, same as KNN)
################################################################################

y_train_num <- ifelse(y_train == "YES", 1, 0)
y_test_num  <- ifelse(y_test  == "YES", 1, 0)

X_train_nn <- as.matrix(X_train_scaled)
X_test_nn  <- as.matrix(X_test_scaled)

nn_model <- nnet(
  x = X_train_nn,
  y = y_train_num,
  size = 10,
  maxit = 200,
  decay = 0.001,
  linout = FALSE,   # classification, not regression
  trace = FALSE
)

# Predicted probabilities
nn_prob <- predict(nn_model, X_test_nn, type = "raw")

# Convert to class labels
nn_pred <- factor(ifelse(nn_prob > 0.5, "YES", "NO"),
                  levels = c("NO", "YES"))





confusionMatrix(nn_pred, y_test)

# AUC
nn_auc <- roc(y_test, as.numeric(nn_prob))
auc(nn_auc)



################################################################################
# MODEL COMPARISON
################################################################################

results <- data.frame(
  Model = c("GLM", "LDA", "KNN (k=5)", "Random Forest", "Neural Network"),
  Accuracy = c(
    confusionMatrix(glm_pred, y_test)$overall["Accuracy"],
    confusionMatrix(lda_pred, y_test)$overall["Accuracy"],
    confusionMatrix(knn_pred, y_test)$overall["Accuracy"],
    confusionMatrix(rf_pred, y_test)$overall["Accuracy"],
    confusionMatrix(nn_pred, y_test)$overall["Accuracy"]
  ),
  AUC = c(
    auc(glm_auc),
    auc(lda_auc),
    NA,                  # KNN has no probability output
    auc(rf_auc),
    auc(nn_auc)
  )
)

print(results)


confusionMatrix(glm_pred, y_test)$byClass[c("Sensitivity","Specificity")]
confusionMatrix(lda_pred, y_test)$byClass[c("Sensitivity","Specificity")]
confusionMatrix(knn_pred, y_test)$byClass[c("Sensitivity","Specificity")]
confusionMatrix(rf_pred, y_test)$byClass[c("Sensitivity","Specificity")]
confusionMatrix(nn_pred, y_test)$byClass[c("Sensitivity","Specificity")]


bal_acc <- function(cm) {
  mean(cm$byClass[c("Sensitivity","Specificity")])
}

bal_acc(confusionMatrix(glm_pred, y_test))
bal_acc(confusionMatrix(lda_pred, y_test))
bal_acc(confusionMatrix(knn_pred, y_test))
bal_acc(confusionMatrix(rf_pred, y_test))
bal_acc(confusionMatrix(nn_pred, y_test))


