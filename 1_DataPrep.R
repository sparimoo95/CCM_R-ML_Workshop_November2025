## CCM Machine Learning in R Workshop: Data Preparation

setwd("/Users/shireenparimoo/Documents/Teaching/R Workshop - November 2025/data/")

#
## 00. Load libraries ----------------------------------------------------

# install.packages("tidyverse", "caret", "pROC", "caTools", "randomForest", "ranger", "tidymodels", "factoextra")
library(tidyverse) # we will use this library, which comes with its own syntax
library(pROC) # plot ROC curves and calculate area under the curve (AUC)
library(caret) # specify models
library(caTools) # split dataset into training and test sets 
library(e1071) # svm
library(randomForest) # random forest model
library(tidymodels)
library(factoextra) # k means viz
library(corrplot) # correlation plot
library(ranger) # random forest using caret

#
## 01. Import data ------------------------------------------------------

raw_heart_df <- read.csv("framingham.csv", header = TRUE, sep = ",")

#

# 02. Prepare and explore the data ----------------------------------------------------

# Framingham Heart Dataset ------------------------------------------------

str(raw_heart_df) # prints out the structure of the dataframe

# prepare the dataset for logistic regression
prepped_heart_df <- raw_heart_df %>%  # this is a pipe; it allows you to perform a sequence of actions on a single object
  # now let's first change some of the variables to factors and numerics
  # the mutate_at function will take a vector of the column names you want to change to factor 
  # and apply the `factor` function to those variables in the `prepped_df_tidy` dataframe
  mutate_at(c("male", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "TenYearCHD"), 
            as.factor) %>% 
  mutate_at(c("age", "cigsPerDay", "totChol", "sysBP", "diaBP", "heartRate", "glucose"),
            as.numeric) %>% 
  # let's also change the values in the male and education columns to make them more descriptive
  # and convert them to factors all at once
  mutate(sex = as.factor((ifelse(male == 1, "M", "F"))),                              
         education = as.factor(case_when(education == 1 ~ "< High School",
                                         education == 2 ~ "High School Graduate",
                                         education == 3 ~ "Some College",
                                         education == 4 ~ "College Graduate"))) %>% 
  # remove rows with missing values
  na.omit() %>% 
  # remove redundant column "male"
  select(-male)

str(prepped_heart_df)

# prepare the dataset for clustering i.e., only keep numeric/continuous variables
prepped_heart_df_cluster <- prepped_heart_df %>% 
  select(c("age", "cigsPerDay", "totChol", "sysBP", "diaBP", "heartRate", "glucose")) %>% 
  # scale and center the numeric variables
  scale() %>%
  # convert to data frame
  as.data.frame() %>% 
  # remove rows with NAs
  na.omit()

# take a look at heart disease rates

plyr::count(prepped_heart_df$TenYearCHD)

# 3099 patients do not develop heart disease, 557 patients do (~15.2%) 

