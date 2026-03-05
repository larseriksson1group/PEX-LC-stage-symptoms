# ML_analysis.R

# Binary classification of lung cancer (advanced and non-advanced) vs controls
# using repeated 10-fold cross-validated Regularized Logistic Regression (RLR), 
# Random Forest (RF), and XGBoost (XGB).

# Performance, calibration, decision curve analysis, and variable importance
# are computed and saved to results/ML_analysis/.

# Functions used for the analysis are defined in ML_functions.R and plotting_functions.R.

library(caret)
library(pROC)
library(ggplot2)
library(openxlsx)
library(MLmetrics)
library(tidyverse)
library(patchwork)
library(RColorBrewer)
library(ggh4x)
library(colorspace)
library(CalibrationCurves)
library(rmda)
library(eulerr)
library(tidytext)

source('code/plotting_functions.R')
source('code/ML_functions.R')

# --------------------------------------------------------------------------- #
# -------------------------------- Run analysis ----------------------------- 
# --------------------------------------------------------------------------- #

response_var <- 'Stage'

## Non-advanced LC vs no-cancer controls
positive_class <- 'Non_Advanced_LC'

data_nonadv <- read.csv2("data/342 non_advanced.csv") |>
  dplyr::select(-Patient) |>
  dplyr::rename(Sex = Gender)
data_nonadv[[response_var]] <- factor(data_nonadv[[response_var]],
                                      levels = c(0, 1),
                                      labels = c("No_Cancer", "Non_Advanced_LC"))

# All predictors (background + symptoms)
res_nonadv <- run_ml_analysis(
  data_nonadv, 
  response_var, 
  positive_class,
  res_path = "results/ML_analysis/non_advanced",
  n_folds = 10, 
  n_repeats = 10, 
  seed = 900, 
  n_cores = 8)

# Background variables only
res_nonadv_bg <- run_ml_analysis(
  data_nonadv |> dplyr::select(1:24),
  response_var, 
  positive_class,
  res_path = "results/ML_analysis/non_advanced_backgr",
  n_folds = 10, 
  n_repeats = 10, 
  seed = 900, # same seed to ensure identical CV folds for comparability
  n_cores = 8)

# Symptom variables only
res_nonadv_symp <- run_ml_analysis(
  data_nonadv |> dplyr::select(c(1, 25:ncol(data_nonadv))),
  response_var, 
  positive_class,
  res_path = "results/ML_analysis/non_advanced_symptoms",
  n_folds = 10, 
  n_repeats = 10, 
  seed = 900, 
  n_cores = 8)

## Advanced LC vs no-cancer controls
positive_class <- 'Advanced_LC'

data_adv <- read.csv2("data/336 advanced.csv") |>
  dplyr::select(-Patient) |>
  dplyr::rename(Sex = Gender)
data_adv[[response_var]] <- factor(data_adv[[response_var]],
                                   levels = c(0, 1),
                                   labels = c("No_Cancer", "Advanced_LC"))

# All predictors (background + symptoms)
res_adv <- run_ml_analysis(
  data_adv, 
  response_var, 
  positive_class,
  res_path = "results/ML_analysis/advanced",
  n_folds = 10, 
  n_repeats = 10, 
  seed = 900, 
  n_cores = 8)

# Background variables only (columns 1-24)
res_adv_bg <- run_ml_analysis(
  data_adv |> dplyr::select(1:24),
  response_var, 
  positive_class,
  res_path = "results/ML_analysis/advanced_backgr",
  n_folds = 10,
  n_repeats = 10, 
  seed = 900, 
  n_cores = 8)

# Symptom variables only (Stage + columns 25 onwards)
res_adv_symp <- run_ml_analysis(
  data_adv |> dplyr::select(c(1, 25:ncol(data_adv))),
  response_var, 
  positive_class,
  res_path = "results/ML_analysis/advanced_symptoms",
  n_folds = 10, 
  n_repeats = 10, 
  seed = 900, 
  n_cores = 8)

# --------------------------------------------------------------------------- #
# ----------------------------- Load results -------------------------------- 
# --------------------------------------------------------------------------- #

load_res <- function(path) {
  list(
    best_params = read.csv(file.path(path, 'best_models_mean_perf.csv')),
    best_param_results = read.csv(file.path(path, 'best_models_all_perf.csv')),
    best_param_preds = read.csv(file.path(path, 'best_models_all_pred.csv')),
    var_imp_perm = read.csv(file.path(path, 'var_imp_permutation.csv')),
    var_imp_model = read.csv(file.path(path, 'var_imp_model_based.csv')),
    var_imp_perm_summary = read.csv(file.path(path, 'var_imp_permutation_summary.csv')),
    var_imp_model_summary = read.csv(file.path(path, 'var_imp_model_based_summary.csv')),
    used_vars = read.csv(file.path(path, 'selected_vars.csv')))
}

res_adv <- load_res('results/ML_analysis/advanced')
res_nonadv <- load_res('results/ML_analysis/non_advanced')

res_adv_bg <- load_res('results/ML_analysis/advanced_backgr')
res_adv_symp <- load_res('results/ML_analysis/advanced_symptoms')
res_nonadv_bg <- load_res('results/ML_analysis/non_advanced_backgr')
res_nonadv_symp <- load_res('results/ML_analysis/non_advanced_symptoms')

# Class label vectors (positive class is always second)
classes_adv <- levels(data_adv[[response_var]])
classes_nonadv <- levels(data_nonadv[[response_var]])

# Verify that all three variable-set analyses used identical CV folds
train_objs_nonadv <- list(
  All = readRDS("results/ML_analysis/non_advanced/train_objects.rds"),
  Background = readRDS("results/ML_analysis/non_advanced_backgr/train_objects.rds"),
  Symptoms = readRDS("results/ML_analysis/non_advanced_symptoms/train_objects.rds"))

train_folds_nonadv <- map(train_objs_nonadv, function(train_list) {
  map(train_list, function(model) model$control$index)
})

stopifnot(all(unlist(train_folds_nonadv$All) == unlist(train_folds_nonadv$Background)))
stopifnot(all(unlist(train_folds_nonadv$All) == unlist(train_folds_nonadv$Symptoms)))

train_objs_adv <- list(
  All = readRDS("results/ML_analysis/advanced/train_objects.rds"),
  Background = readRDS("results/ML_analysis/advanced_backgr/train_objects.rds"),
  Symptoms = readRDS("results/ML_analysis/advanced_symptoms/train_objects.rds"))

train_folds_adv <- map(train_objs_adv, function(train_list) {
  map(train_list, function(model) model$control$index)
})

stopifnot(all(unlist(train_folds_adv$All) == unlist(train_folds_adv$Background)))
stopifnot(all(unlist(train_folds_adv$All) == unlist(train_folds_adv$Symptoms)))

rm(train_objs_nonadv, train_objs_adv, train_folds_nonadv, train_folds_adv)

# --------------------------------------------------------------------------- #
# ----------------------- Performance by model ------------------------------ 
# --------------------------------------------------------------------------- #

# Machine learning performance by model (RLR, RF, XGB) for the analysis with all predictors

# Create and save performance summary for best models
perf_adv <- perf_summary(res_adv)
perf_nonadv <- perf_summary(res_nonadv)

write.csv(perf_adv,
          file = 'results/ML_analysis/advanced/perf_summary_best_models_adv.csv',
          row.names = FALSE)
write.csv(perf_nonadv,
          file = 'results/ML_analysis/non_advanced/perf_summary_best_models_nonadv.csv',
          row.names = FALSE)

# Plot performance metrics

p1 <- plot_perf(perf_nonadv) +
  labs(title = 'Non-advanced lung cancer')

p2 <- plot_perf(perf_adv) +
  labs(title = 'Advanced lung cancer')

(p1 | p2) / guide_area() +
  plot_layout(guides = 'collect', heights = c(1, 0.01)) &
  theme(legend.position = "bottom")

ggsave(filename = "perf_by_model_combined.pdf",
       path = 'results/ML_analysis',
       width = 18, height = 7, units = 'cm')

# --------------------------------------------------------------------------- #
# ------------------------- ROC curves by model -----------------------------
# --------------------------------------------------------------------------- #

# ROC curves for each model (RLR, RF, XGB)
# ROC curves are calculated using the pROC package

# Calculate mean ROC curves across resamples for each model
mean_rocs_adv <- res_adv$best_param_preds |>
  group_by(model) |>
  group_modify(~mean_roc(.x, classes_adv))

mean_rocs_nonadv <- res_nonadv$best_param_preds |>
  group_by(model) |>
  group_modify(~mean_roc(.x, classes_nonadv))

# Plot mean ROC curves
roc_adv <- roc_plot(mean_rocs_adv, perf_adv, classes_adv,
                    'Advanced lung cancer')

roc_nonadv <- roc_plot(mean_rocs_nonadv, perf_nonadv, classes_nonadv,
                       'Non-advanced lung cancer')

(plot_spacer() | roc_nonadv | plot_spacer() | roc_adv) +
  plot_layout(guides = 'collect', widths = c(0.01, 1, 0.01, 1))

ggsave(filename = "ROC_by_model.pdf",
       path = 'results/ML_analysis',
       width = 18, height = 7.5, units = 'cm')


# --------------------------------------------------------------------------- #
# ------------------- Performance at optimal cutoff ------------------------- 
# --------------------------------------------------------------------------- #

# Machine learning performance at optimal probability threshold for each model, 
# determined by maximizing Youden's index (sensitivity + specificity - 1)

# Calculate performance metrics at optimal cutoff for each resample and model
perf_optimal_adv <- res_adv$best_param_preds |>
  group_by(Resample, model) |>
  group_modify(~calc_optimal_cutoff_perf(.x, positive_class = 'Advanced_LC')) |>
  left_join(res_adv$best_param_results |> dplyr::select(Resample, model, ROC),
            by = c('Resample', 'model'))

perf_optimal_nonadv <- res_nonadv$best_param_preds |>
  group_by(Resample, model) |>
  group_modify(~calc_optimal_cutoff_perf(.x, positive_class = 'Non_Advanced_LC')) |>
  left_join(res_nonadv$best_param_results |> dplyr::select(Resample, model, ROC),
            by = c('Resample', 'model'))

# Summarize performance at optimal cutoff and save results
perf_optimal_summary_adv <- summarise_optimal_perf(perf_optimal_adv)
perf_optimal_summary_nonadv <- summarise_optimal_perf(perf_optimal_nonadv)

write.csv(perf_optimal_summary_nonadv,
          file = 'results/ML_analysis/non_advanced/perf_summary_optimal_cutoff_nonadv.csv',
          row.names = FALSE)
write.csv(perf_optimal_summary_adv,
          file = 'results/ML_analysis/advanced/perf_summary_optimal_cutoff_adv.csv',
          row.names = FALSE)

# Save optimal threshold values per model
opt_cutoff_table <- bind_rows(
  'Advanced' = perf_optimal_summary_adv |> filter(metric == 'threshold'),
  'Non_advanced' = perf_optimal_summary_nonadv |> filter(metric == 'threshold'),
  .id = 'Analysis') |>
  mutate(across(mean:ci_upper, ~round(.x, 3)))

write.csv(opt_cutoff_table,
          file = "results/ML_analysis/optimal_cutoffs.csv",
          row.names = FALSE)

# Plot performance metrics at optimal cutoff
p1 <- plot_perf(perf_optimal_summary_nonadv |>
                  filter(!metric %in% c('threshold', 'Accuracy'))) +
  labs(title = 'Model performance (adjusted prob. threshold)',
       subtitle = 'Non-advanced lung cancer')

p2 <- plot_perf(perf_optimal_summary_adv |>
                  filter(!metric %in% c('threshold', 'Accuracy'))) +
  labs(title = 'Model performance (adjusted prob. threshold)',
       subtitle = 'Advanced lung cancer')

(p1 | p2) / guide_area() +
  plot_layout(guides = 'collect', heights = c(1, 0.01)) &
  theme(legend.position = "bottom")

ggsave(filename = "perf_optimal_cutoff_by_model_combined.pdf",
       path = 'results/ML_analysis',
       width = 18, 
       height = 7, 
       units = 'cm')

# --------------------------------------------------------------------------- #
# ------------------------ Calibration assessment --------------------------- 
# --------------------------------------------------------------------------- #

# Calibration curves and stats are calculated using the CalibrationCurves package, which
# implements the val.prob.ci.2 function that computes calibration curves with confidence intervals

## Calibration curves for advanced LC

cal_data_adv <- res_adv$best_param_preds |>
  mutate(obs = factor(obs))

# Calculate calibration curves and stats for each model
cal_res_adv <- map(c('RLR', 'RF', 'XGB'), function(m) {
  dat <- cal_data_adv |>
    filter(model == m, !is.na(Advanced_LC)) |>
    mutate(obs = ifelse(obs == 'Advanced_LC', 1, 0))
  val.prob.ci.2(p = dat[['Advanced_LC']], y = dat$obs, logit = 'p', pl = TRUE,
                logistic.cal = TRUE)
})
names(cal_res_adv) <- c('RLR', 'RF', 'XGB')

# Extract calibration curve data (LOESS smoothed)
cal_df_adv <- map(cal_res_adv, ~.x$CalibrationCurves$FlexibleCalibration) |>
  bind_rows(.id = 'model')

# Extract calibration statistics (intercept and slope, logistic calibration model)
cal_stats_adv <- map(cal_res_adv, ~.x$stats[c('Intercept', 'Slope')]) |>
  bind_rows(.id = 'model')


## Calibration curves for non-advanced LC

cal_data_nonadv <- res_nonadv$best_param_preds |>
  mutate(obs = factor(obs))

# Calculate calibration curves and stats for each model
cal_res_nonadv <- map(c('RLR', 'RF', 'XGB'), function(m) {
  dat <- cal_data_nonadv |>
    filter(model == m, !is.na(Non_Advanced_LC)) |>
    mutate(obs = ifelse(obs == 'Non_Advanced_LC', 1, 0))
  val.prob.ci.2(p = dat[['Non_Advanced_LC']], y = dat$obs, logit = 'p', pl = TRUE,
                logistic.cal = TRUE)
})
names(cal_res_nonadv) <- c('RLR', 'RF', 'XGB')

# Extract calibration curve data and stats (LOESS smoothed)
cal_df_nonadv <- map(cal_res_nonadv, ~.x$CalibrationCurves$FlexibleCalibration) |>
  bind_rows(.id = 'model')

# Extract calibration statistics (intercept and slope, logistic calibration model)
cal_stats_nonadv <- map(cal_res_nonadv, ~.x$stats[c('Intercept', 'Slope')]) |>
  bind_rows(.id = 'model')


## Plot calibration curves for both analyses

(cal_plot(cal_df_nonadv, cal_stats_nonadv, 'Non-advanced lung cancer') /
    cal_plot(cal_df_adv, cal_stats_adv, 'Advanced lung cancer')) /
  guide_area() +
  plot_layout(guides = 'collect', heights = c(1, 1, 0.01)) &
  theme(legend.position = 'bottom')

ggsave(filename = "calibration_curves.pdf",
       path = 'results/ML_analysis',
       width = 16.5, 
       height = 15, 
       units = 'cm')

# --------------------------------------------------------------------------- #
# ----------------------- Decision curve analysis --------------------------- 
# --------------------------------------------------------------------------- #

# Decision curve analysis is performed using the rmda package

prep_dca_data <- function(best_param_preds, positive_class, classes) {
  # Pivot predictions wide so each model has its own column, as required by rmda
  best_param_preds |>
    dplyr::select(Resample, rowIndex, model, obs, all_of(classes[2])) |>
    mutate(obs = ifelse(obs == classes[2], 1, 0)) |>
    pivot_wider(names_from = model, values_from = all_of(classes[2]))
}

run_dca <- function(dca_data, formula_str) {
  decision_curve(
    as.formula(formula_str),
    data = dca_data,
    thresholds = seq(0, 1, by = 0.01),
    fitted.risk = TRUE) # use predicted probabilities
}

format_dca_results <- function(dca_list) {
  # Combine results and standardize model labels
  bind_rows(dca_list, .id = 'Model') |>
    mutate(
      model = case_when(
        model == 'obs ~ RLR'  ~ 'RLR',
        model == 'obs ~ RF'  ~ 'RF',
        model == 'obs ~ XGB' ~ 'XGB',
        TRUE ~ model),
      model = factor(model, levels = c('None', 'All', 'RLR', 'RF', 'XGB')))
}

## DCA for advanced LC

dca_data_adv <- prep_dca_data(res_adv$best_param_preds, 'Advanced_LC', classes_adv)
dca_res_adv  <- format_dca_results(list(
  RLR = run_dca(dca_data_adv, 'obs ~ RLR')$derived.data,
  RF = run_dca(dca_data_adv, 'obs ~ RF')$derived.data,
  XGB = run_dca(dca_data_adv, 'obs ~ XGB')$derived.data))


## DCA for non-advanced LC

dca_data_nonadv <- prep_dca_data(res_nonadv$best_param_preds, 'Non_Advanced_LC', classes_nonadv)
dca_res_nonadv  <- format_dca_results(list(
  RLR = run_dca(dca_data_nonadv, 'obs ~ RLR')$derived.data,
  RF = run_dca(dca_data_nonadv, 'obs ~ RF')$derived.data,
  XGB = run_dca(dca_data_nonadv, 'obs ~ XGB')$derived.data))


## Plot DCA curves for both analyses

dca_plot(dca_res_nonadv, 'Non-advanced lung cancer') +
  dca_plot(dca_res_adv,    'Advanced lung cancer')

ggsave(filename = "dca_plots.pdf",
       path = 'results/ML_analysis',
       width = 18, 
       height = 6.5, 
       units = 'cm')

# --------------------------------------------------------------------------- #
# ----------------- Performance by predictor subset ------------------------- 
# --------------------------------------------------------------------------- #

# Compare performance between models trained on all predictors vs background-only vs symptom-only

perf_nonadv_bg <- perf_summary(res_nonadv_bg)
perf_nonadv_symp <- perf_summary(res_nonadv_symp)
perf_adv_bg <- perf_summary(res_adv_bg)
perf_adv_symp <- perf_summary(res_adv_symp)

perf_nonadv_by_varset <- bind_rows(
  perf_nonadv |> mutate(Variable_Set = 'All'),
  perf_nonadv_bg |> mutate(Variable_Set = 'Background'),
  perf_nonadv_symp |> mutate(Variable_Set = 'Symptoms'))

perf_adv_by_varset <- bind_rows(
  perf_adv |> mutate(Variable_Set = 'All'),
  perf_adv_bg |> mutate(Variable_Set = 'Background'),
  perf_adv_symp |> mutate(Variable_Set = 'Symptoms'))

plot_perf2(perf_nonadv_by_varset) +
  labs(subtitle = 'Non-advanced lung cancer')

ggsave('perf_by_variable_set_non_advanced.pdf',
       path = 'results/ML_analysis',
       width = 18, height = 8, units = 'cm')

plot_perf2(perf_adv_by_varset) +
  labs(subtitle = 'Advanced lung cancer')

ggsave('perf_by_variable_set_advanced.pdf',
       path = 'results/ML_analysis',
       width = 18, height = 8, units = 'cm')

# --------------------------------------------------------------------------- #
# ----------------- ROC curves by predictor subset -------------------------
# --------------------------------------------------------------------------- #

# ROC curves for each model (RLR, RF, XGB) by variable set (all predictors vs background-only vs symptom-only)

# Prepare predictions for ROC curve calculation
preds_nonadv_varset <- list(
  All = res_nonadv$best_param_preds,
  Background = res_nonadv_bg$best_param_preds,
  Symptoms = res_nonadv_symp$best_param_preds)

preds_adv_varset <- list(
  All = res_adv$best_param_preds,
  Background = res_adv_bg$best_param_preds,
  Symptoms = res_adv_symp$best_param_preds)

# Calculate mean ROC curves across resamples for each model and variable set
mean_rocs_nonadv_varset <- map(preds_nonadv_varset, function(preds) {
  preds |> group_by(model) |> group_modify(~mean_roc(.x, classes_nonadv))
})

mean_rocs_adv_varset <- map(preds_adv_varset, function(preds) {
  preds |> group_by(model) |> group_modify(~mean_roc(.x, classes_adv))
})

# Plot mean ROC curves for both analyses
p1 <- roc_plot2(mean_rocs_nonadv_varset, perf_nonadv_by_varset,
                classes_nonadv, 'Non-advanced lung cancer')

p2 <- roc_plot2(mean_rocs_adv_varset, perf_adv_by_varset,
                classes_adv, 'Advanced lung cancer')

p1 / p2 / guide_area() +
  plot_layout(guides = 'collect', heights = c(1, 1, 0.01)) &
  theme(legend.position = "bottom")

ggsave(filename = "ROC_by_variable_set.pdf",
       path = 'results/ML_analysis',
       width = 16.5, 
       height = 15, 
       units = 'cm')

# --------------------------------------------------------------------------- #
# ------------------------- Variable importance ----------------------------- 
# --------------------------------------------------------------------------- #

# Variable importance is calculated using both permutation importance (mean 
# decrease in ROC-AUC when variable values are randomly permuted) and model-based 
# importance (tree-level permutation importance for RF, absolute coefficient values for RLR,
# gain importance for XGB).


## Variable importance summarized over folds

# Combine variable importance summaries for both analyses and modify variable names for plotting
var_imp_summary_all <- list(
  Advanced = res_adv$var_imp_perm_summary |> mutate(Variable_name = create_display_names(Variable)),
  Non_advanced = res_nonadv$var_imp_perm_summary |> mutate(Variable_name = create_display_names(Variable)))

var_imp_model_summary_all <- list(
  Advanced = res_adv$var_imp_model_summary |> mutate(Variable_name = create_display_names(Variable)),
  Non_advanced = res_nonadv$var_imp_model_summary |> mutate(Variable_name = create_display_names(Variable)))

# Merge permutation and model-based importance; scale permutation importance to 0-1
var_imp_summary_merged <- map2(
  var_imp_summary_all, var_imp_model_summary_all,
  function(perm_df, model_df) {
    perm_df |>
      full_join(model_df, by = c('model', 'Variable', 'Variable_name'),
                suffix = c('_perm', '_model')) |>
      group_by(model) |>
      mutate(importance_scaled_perm =
               (mean_importance_perm - min(mean_importance_perm)) /
               (max(mean_importance_perm) - min(mean_importance_perm))) |>
      dplyr::rename(importance_scaled_model = importance_scaled) |>
      ungroup()
  })

# Rank the top variables by all six importance metrics and compute a median rank
ranked_vars <- map(var_imp_summary_merged, rank_all_metrics)

# Select top 20 variables by median rank across all six importance metrics
top_vars_median_rank <- map(ranked_vars, function(df) {
  df |>
    group_by(model) |>
    slice_min(order_by = tibble(median_rank_metrics, mean_importance_model), n = 20) |>
    dplyr::select(model, Variable) |>
    mutate(Order = row_number())
})

## Importance heatmaps

plot_imp_heatmap(var_imp_summary_merged$Advanced,
                 top_vars_median_rank$Advanced,
                 'Advanced lung cancer')

ggsave(filename = "var_imp_heatmap_advanced.pdf",
       path = 'results/ML_analysis',
       width = 18, 
       height = 7.8, 
       units = 'cm')

plot_imp_heatmap(var_imp_summary_merged$Non_advanced,
                 top_vars_median_rank$Non_advanced,
                 'Non-advanced lung cancer')

ggsave(filename = "var_imp_heatmap_non_advanced.pdf",
       path = 'results/ML_analysis',
       width = 18, 
       height = 7.8, 
       units = 'cm')

## Variable importance stability over folds (boxplots)

# Combine variable importance results per fold for both analyses
# and modify variable names for plotting
var_imp_perm_all <- list(
  Advanced = res_adv$var_imp_perm,
  Non_advanced = res_nonadv$var_imp_perm)

var_imp_model_all <- list(
  Advanced = res_adv$var_imp_model,
  Non_advanced = res_nonadv$var_imp_model)

var_imp_perm_all <- map(var_imp_perm_all, ~mutate(.x, Variable_name = create_display_names(Variable)))
var_imp_model_all <- map(var_imp_model_all, ~mutate(.x, Variable_name = create_display_names(Variable)))

# Plot boxplots
set.seed(123)
var_imp_boxplot(var_imp_perm_all$Non_advanced, top_vars_median_rank$Non_advanced,
                'Non-advanced lung cancer', 'perm') /
  var_imp_boxplot(var_imp_perm_all$Advanced, top_vars_median_rank$Advanced,
                  'Advanced lung cancer', 'perm') /
  guide_area() +
  plot_layout(guides = 'collect', heights = c(1, 1, 0.01))

ggsave(filename = "var_imp_perm_boxplot.pdf",
       path = 'results/ML_analysis',
       width = 16.5, 
       height = 15, 
       units = 'cm', 
       device = cairo_pdf)

var_imp_boxplot(var_imp_model_all$Non_advanced, top_vars_median_rank$Non_advanced,
                'Non-advanced lung cancer', 'model') /
  var_imp_boxplot(var_imp_model_all$Advanced, top_vars_median_rank$Advanced,
                  'Advanced lung cancer', 'model') /
  guide_area() +
  plot_layout(guides = 'collect', heights = c(1, 1, 0.01))

ggsave(filename = "var_imp_model_boxplot.pdf",
       path = 'results/ML_analysis',
       width = 16.5, 
       height = 15, 
       units = 'cm', 
       device = cairo_pdf)

# --------------------------------------------------------------------------- #
# ------------ Overlap of top variables between models ------ --------------- 
# --------------------------------------------------------------------------- #

# Plot venn diagrams showing the overlap between the top 20 variables of each model
# Proportional Venn diagrams are plotted using the eulerr package

venn_data_adv <- top_vars_median_rank$Advanced |>
  group_by(model) |>
  summarise(Variables = list(Variable)) |>
  deframe()

venn_data_nonadv <- top_vars_median_rank$Non_advanced |>
  group_by(model) |>
  summarise(Variables = list(Variable)) |>
  deframe()

pdf('results/ML_analysis/top_vars_venn_advanced.pdf', width = 4 / 2.54, height = 4 / 2.54)
plot(euler(venn_data_adv),
     fills = lighten(colors, 0.4),
     edges = colors,
     quantities = list(cex = 0.5),
     main = list(label = 'Advanced lung cancer', cex = 0.67),
     labels = list(cex = 0.5))
dev.off()

pdf('results/ML_analysis/top_vars_venn_non_advanced.pdf', width = 4 / 2.54, height = 4 / 2.54)
plot(euler(venn_data_nonadv),
     fills = lighten(colors, 0.4),
     edges = colors,
     quantities = list(cex = 0.5),
     main = list(label = 'Non-advanced lung cancer', cex = 0.67),
     labels = list(cex = 0.5))
dev.off()

# Save which variables appear in each section of each Venn diagram
save_intersections <- function(venn_data, path) {
  intersections <- list(
    RLR_only = setdiff(setdiff(venn_data$RLR, venn_data$RF), venn_data$XGB),
    RF_only = setdiff(setdiff(venn_data$RF, venn_data$RLR), venn_data$XGB),
    XGB_only = setdiff(setdiff(venn_data$XGB, venn_data$RLR), venn_data$RF),
    RLR_RF = setdiff(intersect(venn_data$RLR, venn_data$RF), venn_data$XGB),
    RLR_XGB = setdiff(intersect(venn_data$RLR, venn_data$XGB), venn_data$RF),
    RF_XGB = setdiff(intersect(venn_data$RF, venn_data$XGB), venn_data$RLR),
    All_three = intersect(intersect(venn_data$RLR, venn_data$RF), venn_data$XGB))
  sink(path)
  for (name in names(intersections)) {
    cat("\n", name, ":\n", sep = "")
    cat(intersections[[name]], sep = ", ")
    cat("\n")
  }
  sink()
}

save_intersections(venn_data_adv,
                   "results/ML_analysis/top_vars_intersections_adv.txt")
save_intersections(venn_data_nonadv,
                   "results/ML_analysis/top_vars_intersections_nonadv.txt")

# --------------------------------------------------------------------------- #
# ------ Overlap of top variables between advanced and non-advanced --------- 
# --------------------------------------------------------------------------- #

# Compare the top variables between the advanced and non-advanced analyses
# (= variables shown in Figure 3c)

# Variables shared across all three models in both analyses
cat("Shared across all models and both analyses:\n")
both <- intersect(
  intersect(intersect(venn_data_adv$RF, venn_data_adv$RLR), venn_data_adv$XGB),
  intersect(intersect(venn_data_nonadv$RF, venn_data_nonadv$RLR), venn_data_nonadv$XGB))
print(both)

cat("Shared across all models in non-advanced only:\n")
non_adv_only <- setdiff(
  intersect(intersect(venn_data_nonadv$RF, venn_data_nonadv$RLR), venn_data_nonadv$XGB),
  intersect(intersect(venn_data_adv$RF, venn_data_adv$RLR), venn_data_adv$XGB))
print(non_adv_only)

cat("Shared across all models in advanced only:\n")
adv_only <- setdiff(
  intersect(intersect(venn_data_adv$RF, venn_data_adv$RLR), venn_data_adv$XGB),
  intersect(intersect(venn_data_nonadv$RF, venn_data_nonadv$RLR), venn_data_nonadv$XGB))
print(adv_only)

# Save variables in one file
intersections <- list(
  Both = both,
  `Non-advanced only` = non_adv_only,
  `Advanced only` = adv_only)
sink("results/ML_analysis/top_vars_all_models_adv_vs_nonadv.txt")
for (name in names(intersections)) {
  cat("\n", name, ":\n", sep = "")
  cat(intersections[[name]], sep = ", ")
  cat("\n")
}
sink()

# Per-model comparison of top variables between advanced and non-advanced analyses
top_vars_adv_vs_nonadv <- list(
  RLR = list(
    `Non-advanced` = top_vars_median_rank$Non_advanced |> filter(model == 'RLR') |> pull(Variable),
    Advanced = top_vars_median_rank$Advanced |> filter(model == 'RLR') |> pull(Variable)),
  RF = list(
    `Non-advanced` = top_vars_median_rank$Non_advanced |> filter(model == 'RF') |> pull(Variable),
    Advanced = top_vars_median_rank$Advanced |> filter(model == 'RF') |> pull(Variable)),
  XGB = list(
    `Non-advanced` = top_vars_median_rank$Non_advanced |> filter(model == 'XGB') |> pull(Variable),
    Advanced = top_vars_median_rank$Advanced |> filter(model == 'XGB') |> pull(Variable)))

# Save in separate files per model
for (algorithm in names(top_vars_adv_vs_nonadv)) {

  sink(paste0("results/ML_analysis/top_vars_intersections_", algorithm,
              "_adv_vs_nonadv.txt"))
  for (name in names(intersections)) {
    cat("\n", name, ":\n", sep = "")
    cat(intersections[[name]], sep = ", ")
    cat("\n")
  }
  sink()
}

# --------------------------------------------------------------------------- #
# ---------------------- Number of selected variables ----------------------- 
# --------------------------------------------------------------------------- #

all_selected_vars <- bind_rows(
  'Advanced' = res_adv$used_vars,
  'Non_advanced' = res_nonadv$used_vars,
  .id = 'Analysis') |>
  mutate(Analysis = factor(Analysis,
                           levels = c('Non_advanced', 'Advanced'),
                           labels = c('Non-advanced lung cancer', 'Advanced lung cancer')))

mean_df <- all_selected_vars |>
  group_by(Analysis, model) |>
  summarise(mean_n = round(mean(n_variables)), .groups = "drop")

all_selected_vars |>
  ggplot(aes(x = n_variables, fill = model)) +
  geom_histogram(binwidth = 1, position = 'identity', color = 'grey30', alpha = 0.6) +
  geom_vline(data = mean_df,
             aes(xintercept = mean_n, color = model),
             linetype = 'dashed', size = 0.5, show.legend = FALSE) +
  facet_wrap(~Analysis) +
  labs(x = 'Number of selected variables', y = 'Count', fill = 'Model') +
  theme_publ_bw() +
  scale_fill_manual(values = colors, aesthetics = c('color', 'fill')) +
  theme(legend.position = 'bottom') +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  scale_x_continuous(breaks = seq(0, max(all_selected_vars$n_variables) + 5, by = 5))

ggsave(filename = "selected_variables_histogram.pdf",
       path = 'results/ML_analysis',
       width = 16.5, 
       height = 7.5,
       units = 'cm')

# --------------------------------------------------------------------------- #
# -------------------- Additional result tables ----------------------------- 
# --------------------------------------------------------------------------- #

# Best hyperparameters and corresponding mean performance
best_perf_adv <- read.csv("results/ML_analysis/advanced/mean_perf_by_model.csv") |>
  semi_join(res_adv$best_params) |>
  left_join(perf_adv |> filter(metric == 'ROC') |> dplyr::select(model, ci_lower, ci_upper),
            by = 'model') |> 
  dplyr::select(model,
                AUC_mean = ROC_mean, AUC_sd = ROC_sd,
                AUC_CI_lower = ci_lower, AUC_CI_upper = ci_upper,
                alpha, lambda, mtry, splitrule, min.node.size,
                eta, max_depth, gamma, colsample_bytree,
                min_child_weight, subsample, nrounds) |>
  mutate(across(contains('AUC'), ~round(.x, 3)),
         Stage_model = 'Advanced stage')

best_perf_nonadv <- read.csv("results/ML_analysis/non_advanced/mean_perf_by_model.csv") |>
  semi_join(res_nonadv$best_params) |>
  left_join(perf_nonadv |> filter(metric == 'ROC') |> dplyr::select(model, ci_lower, ci_upper),
            by = 'model') |>
  dplyr::select(model,
                AUC_mean = ROC_mean, AUC_sd = ROC_sd,
                AUC_CI_lower = ci_lower, AUC_CI_upper = ci_upper,
                alpha, lambda, mtry, splitrule, min.node.size,
                eta, max_depth, gamma, colsample_bytree,
                min_child_weight, subsample, nrounds) |>
  mutate(across(contains('AUC'), ~round(.x, 3)),
         Stage_model = 'Non-advanced stage')

bind_rows(best_perf_adv, best_perf_nonadv) |>
  dplyr::select(Stage_model, everything()) |>
  write.csv(file = "results/ML_analysis/best_models_perf_and_tuning.csv",
            row.names = FALSE)

# Save session info
writeLines(capture.output(sessionInfo()),
           con = "results/ML_analysis/session_info.txt")

