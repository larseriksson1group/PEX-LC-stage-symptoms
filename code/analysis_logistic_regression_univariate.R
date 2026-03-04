###########################################
# Univariate Logistic Regression Analysis
# ---------------------------------------
# This script reads an Excel dataset, runs
# univariate logistic regressions for each
# independent variable, and saves results
# to an Excel file.
#
# Author: Tina Gustavell
# Date: 2025-02-10
###########################################

# Install required packages (if not installed)
# install.packages(c("readxl", "broom", "dplyr", "writexl"))

# Load packages
library(readxl)
library(broom)
library(dplyr)
library(writexl)

###########################################
# Load data
###########################################

data <- read_excel("PATH_TO_FILE.xlsx") 

###########################################
# Variable definitions
###########################################

# Assumption: the dependent variable is the 2nd column (binary)
dependent_variable <- data[[2]]

# Independent variables: columns 3 to 144
independent_variables <- data[, 3:144]

# Convert all independent variables to numeric
independent_variables <- data.frame(lapply(independent_variables, as.numeric))

###########################################
# Run univariate logistic regressions
###########################################

results_list <- list()

for (i in 1:ncol(independent_variables)) {
  
  # Build formula: dependent_variable ~ predictor
  formula <- as.formula(
    paste("dependent_variable ~ `", names(independent_variables)[i], "`", sep = "")
  )
  
  # Fit logistic regression model
  model <- glm(formula, data = data, family = binomial)
  
  # Extract model summary using broom
  model_summary <- tidy(model, conf.int = TRUE)
  
  # Split dataset into outcome groups 0 and 1
  group_0 <- data[data[[2]] == 0, ]
  group_1 <- data[data[[2]] == 1, ]
  
  # Extract predictor values for each group
  var_0 <- group_0[[i + 2]]   # +2 because predictor starts at column 3
  var_1 <- group_1[[i + 2]]
  
  # Summaries
  Sum_0_1 <- sum(var_0, na.rm = TRUE)
  Sum_1_1 <- sum(var_1, na.rm = TRUE)
  
  # Percentages
  Percent_0_1 <- ifelse(nrow(group_0) > 0,
                        round(100 * Sum_0_1 / nrow(group_0), 2), NA)
  
  Percent_1_1 <- ifelse(nrow(group_1) > 0,
                        round(100 * Sum_1_1 / nrow(group_1), 2), NA)
  
  # Save results for this variable
  results_list[[i]] <- data.frame(
    Variable   = names(independent_variables)[i],
    Sum_0_1    = Sum_0_1,
    Sum_1_1    = Sum_1_1,
    Percent_0_1 = Percent_0_1,
    Percent_1_1 = Percent_1_1,
    Odds_Ratio  = round(exp(model_summary$estimate[2]), 2),
    CI_Lower    = round(exp(model_summary$conf.low[2]), 2),
    CI_Upper    = round(exp(model_summary$conf.high[2]), 2),
    P_value     = round(model_summary$p.value[2], 4)
  )
}

###########################################
# Combine and export results
###########################################

final_results <- bind_rows(results_list)

# Export to Excel
write_xlsx(final_results, "univariate_logistic_regression_results.xlsx")

cat("Results saved to 'univariate_logistic_regression_results.xlsx'\n")

###########################################
# Descriptive statistics: Age by Stage
###########################################

summary_stats <- data %>%
  group_by(Stage) %>%
  summarise(
    mean_age = mean(Age, na.rm = TRUE),
    min_age  = min(Age, na.rm = TRUE),
    max_age  = max(Age, na.rm = TRUE),
    sd_age   = sd(Age, na.rm = TRUE),
    n        = n()
  )

print(summary_stats)

###########################################
# Logistic regression: Stage ~ Age
###########################################

model <- glm(Stage ~ Age, data = data, family = binomial)

model_results <- tidy(model, conf.int = TRUE, exponentiate = TRUE) %>%
  select(term, estimate, conf.low, conf.high, p.value) %>%
  rename(
    Odds_Ratio = estimate,
    Lower_CI   = conf.low,
    Upper_CI   = conf.high,
    P_Value    = p.value
  )

print(model_results)