## CCM Machine Learning in R Workshop: Supervised Learning - Logistic Regression

# 01. Logistic Regression: Heart Disease ---------------------------------------

## GOAL: predict whether someone will develop heart disease (TenYearCHD)

# 0. set seed for reproducibility
set.seed(66)

# 1. split up the dataset into a training and test set
split_heart = sample.split(prepped_heart_df$TenYearCHD, .75)

train_heart = prepped_heart_df[split_heart, ]
test_heart  = prepped_heart_df[!split_heart, ]

# 2. Build a linear model on the training set only - many ways to do this as well
heart_glm = glm(TenYearCHD ~ ., family = "binomial", data = train_heart)
summary(heart_glm)

# 3. Apply the model to the test set and generate the predictions
heart_glm_pred = predict(heart_glm, newdata = test_heart, type = "response")
heart_glm_pred

# 4. Classification accuracy
# if predicted value is greater than 0.5, then set the prediction to "1" i.e., will develop heart disease 
heart_glm_perf = ifelse(heart_glm_pred > 0.5, 1, 0)

# create confusion matrix from model's performance and compare it to the test df 
confusionMatrix(factor(heart_glm_perf), 
                factor(test_heart$TenYearCHD), 
                positive = as.character(1))

# 5 Evaluate variable importance using the varImp function from the caret package
heart_glm_importances <- varImp(heart_glm) %>%
  as.data.frame() %>% 
  rownames_to_column() %>%
  dplyr::rename(predictor = rowname) %>% 
  arrange(desc(Overall)) %>% # arrange variables according to their importance
  top_n(5) %>% # select top 5 most important variables
  # convert predictor to factor
  mutate(predictor = as.factor(predictor))

# clean up the naming of the predictors 
levels(heart_glm_importances$predictor) = c("Age", "HS Education", "Blood Glucose", "Male Sex", "Systolic BP")

# 6. Visualize most important variables
ggplot(heart_glm_importances, 
       aes(x = predictor, y = Overall, fill = predictor)) +
  geom_col(show.legend = F, color = "black") + 
  labs(x = "Variable", y = "Estimate") +
  scale_fill_brewer(palette = 'Set3') + 
  theme_classic(base_size = 24) 

# 7. Plot the ROC curve and estimate the AUC
heart_glm_roc = roc(test_heart$TenYearCHD, heart_glm_pred)
heart_glm_roc
# model has a ~73% probability of correctly distinguishing between a positive and a negative instance
plot(heart_glm_roc)

## CHALLENGE: do the most important predictors change with a different training/test split?
## CHALLENGE: how does the AUC change with a different training/test split? 

#
# 02. Random Forest - Classification --------------------------------------------------------

set.seed(66)

split_heart_rf = sample.split(prepped_heart_df$TenYearCHD, .5)

train_heart_rf = prepped_heart_df[split_heart_rf, ]
test_heart_rf  = prepped_heart_df[!split_heart_rf, ]

heart_rf = randomForest(TenYearCHD ~ ., data = train_heart_rf, 
                        importance = T,  # assesses importance of each predictor for accuracy
                        ntree = 500, # start with 500 trees (default setting if you don't specify it)
                        mtry = 10) # number of predictors randomly selected for each split in the tree
heart_rf

## OUTPUT: includes a confusion matrix, out-of-bag (OOB) error i.e., misclassification rate
## number of trees (you can specify this as well) and number of variables randomly sampled as candidates
## to split by at each split

predict(heart_rf, newdata = test_heart_rf)
heart_rf_pred = predict(heart_rf, newdata = test_heart_rf, type = 'response')

# plot variable importance
varImpPlot(heart_rf, 
           main = "Variable Importance for Predicting Heart Disease", 
           bg = "brown")

## mean decrease accuracy = how much accuracy would decrease if that variable is removed from the model
## mean decrease gini = how often a randomly chosen element of a set would be incorrectly labeled if it were labeled randomly

## is the output from random forest classification similar to logistic regression?
## which variables are most important for predicting heart disease? 
## how does the output change with a different training/test split?

#
# 03. Random Forest - Regression --------------------------------------------------------

## GOAL: predict cholesterol levels

# 0. set seed for reproducibility
set.seed(66)

# 1. split up the dataset into a training and test set
split_heart_rf_linear = sample.split(prepped_heart_df$sysBP, .5)

train_heart_rf_linear = prepped_heart_df[split_heart_rf_linear, ]
test_heart_rf_linear  = prepped_heart_df[!split_heart_rf_linear, ]

# 2. Build a random forest model to predict cholesterol levels
heart_rf_chol = randomForest(totChol ~ ., 
                             data = train_heart_rf_linear,
                             importance = T, 
                             ntree = 500,
                             mtry = 10)
heart_rf_chol

heart_rf_chol_pred = predict(heart_rf_chol, newdata = test_heart_rf_linear, 'response')

# plot variable importance
varImpPlot(heart_rf_chol, 
           main = "Variable Importance for Predicting Cholesterol Levels", 
           bg = "brown")
## %Increase in MSE = how much the mean squared error would increase if a variable were omitted from the model

