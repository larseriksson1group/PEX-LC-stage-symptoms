# ML_functions.R
# Functions for repeated cross-validated ML analysis, performance evaluation,
# and variable importance computation

library(ranger)
library(caret)
library(pROC)
library(glmnet)
library(doParallel)
library(tidyverse)
library(recipes)
library(yardstick)
library(xgboost)
library(future)
library(foreach)
library(doFuture)

# --------------------------------------------------------------------------- #
# ----------------------- Main ML analysis function ------------------------- 
# --------------------------------------------------------------------------- #

run_ml_analysis <- function(data, response_var, positive_class, res_path,
                            output_suffix = NULL, n_folds = 10,
                            n_repeats = 5, seed = 900, n_cores = 4) {
  # Perform repeated k-fold cross-validated ML analysis with hyperparameter tuning.
  # Trains Regularized Logistic Regression (RLR), Random Forest (RF), and XGBoost (XGB)
  # using repeated k-fold CV with grid search. Also fits final per-fold models with
  # best hyperparameters and computes permutation variable importance.
  # All results are written to disk under res_path.
  #
  # data: data frame with predictor and response variables
  # response_var: name of the response variable (string)
  # positive_class: name of the positive class in the response variable (string)
  # res_path: directory path for saving results
  # output_suffix: optional suffix appended to output file names
  # n_folds: number of folds for cross-validation (default: 10)
  # n_repeats: number of repeats for cross-validation (default: 5)
  # seed: random seed for reproducibility (default: 900)
  # n_cores: number of cores for parallel processing (default: 4)
  # Returns a named list with best parameters, performance, predictions, and
  #   variable importance data frames.

  # ----- Setup and data preparation ----
  
  if (!dir.exists(res_path)) {
    dir.create(res_path, recursive = TRUE)
  }

  set.seed(seed)

  data[[response_var]] <- relevel(as.factor(data[[response_var]]), ref = positive_class)

  predictor_vars <- colnames(data)[colnames(data) != response_var]
  n_features <- length(predictor_vars)

  cat("\n=== Dataset Information ===\n")
  cat("Number of samples:", nrow(data), "\n")
  cat("Number of features:", n_features, "\n")
  cat("Classes:", paste(levels(data[[response_var]]), collapse = ", "), "\n")
  cat("Positive class:", positive_class, "\n\n")

  # Normalise Age if present; all other variables passed through unchanged
  train_recipe <- recipe(data) %>%
    update_role(-all_of(response_var), new_role = "predictor") %>%
    update_role(all_of(response_var), new_role = "outcome") %>%
    step_normalize(any_of('Age'))

  # Hyperparameter grids
  rf_grid <- expand.grid(
    mtry = sort(unique(
      pmax(1, round(c(sqrt(n_features), n_features / 10, n_features / 5,
                      n_features / 3, n_features / 2, n_features))))),
    splitrule = "gini",
    min.node.size = c(1, 3, 5, 10, 15, 20, 30))

  glmnet_grid <- expand.grid(
    alpha = seq(0, 1, by = 0.1),
    lambda = 10^seq(-3, 0, length.out = 20))

  xgb_grid <- expand.grid(
    nrounds = c(50, 100, 200),
    max_depth = c(3, 6, 9),
    eta = c(0.01, 0.1, 0.3),
    gamma = c(0, 1, 5),
    colsample_bytree = c(0.6, 0.9),
    min_child_weight = c(1, 3),
    subsample = c(0.5, 0.8))

  cl <- makeCluster(n_cores)
  registerDoParallel(cl)

  set.seed(seed)
  folds <- createMultiFolds(data[[response_var]], k = n_folds, times = n_repeats)

  # Custom summary function returning multiple performance metrics per fold
  comprehensiveSummary <- function(data, lev = NULL, model = NULL) {

    requireNamespace("dplyr")
    requireNamespace("yardstick")

    if (!all(levels(data[, "pred"]) == lev)) {
      stop("levels of observed and predicted data do not match")
    }

    pos_class <- lev[1]

    metrics <- yardstick::metric_set(
      yardstick::roc_auc,
      yardstick::sens,
      yardstick::spec,
      yardstick::pr_auc,
      yardstick::accuracy,
      yardstick::bal_accuracy,
      yardstick::f_meas,
      yardstick::precision,
      yardstick::ppv,
      yardstick::npv,
      yardstick::kap,
      yardstick::mcc)

    results <- data |>
      dplyr::mutate(truth = factor(obs, levels = lev),
                    estimate = factor(pred, levels = lev),
                    .pred_pos = .data[[pos_class]]) |>
      metrics(truth = truth, estimate = estimate, .pred_pos)

    out <- setNames(results$.estimate, results$.metric)

    name_map <- c(
      sens = "Sens",
      spec = "Spec",
      accuracy = "Accuracy",
      bal_accuracy = "BalancedAccuracy",
      f_meas = "F1",
      precision = "Precision",
      ppv = "PPV",
      npv = "NPV",
      kap = "Kappa",
      mcc = "MCC",
      roc_auc = "ROC",
      pr_auc = "PR_AUC")

    names(out) <- name_map[names(out)]
    return(out)
  }

  # Seeds list: length n_folds * n_repeats + 1, each vector of length 1000
  # (length 1000 ensures enough seeds regardless of tuning grid size)
  set.seed(seed)
  seeds <- vector(mode = 'list', length = n_folds * n_repeats + 1)
  for (j in seq_along(seeds)) seeds[[j]] <- sample.int(n = 10000, 1000)

  train_control <- trainControl(
    method = 'repeatedcv',
    index = folds,
    classProbs = TRUE,
    savePredictions = 'all',
    returnResamp = 'all',
    returnData = FALSE,
    summaryFunction = comprehensiveSummary,
    seeds = seeds)

  # ----- Train models -----
  
  models <- vector(mode = 'list', length = 3)
  names(models) <- c('RLR', 'RF', 'XGB')

  message("=== Training Elastic Net with ", n_folds, " fold CV repeated ",
          n_repeats, " times ===")
  message("Testing ", nrow(glmnet_grid), " hyperparameter combinations...")
  models[['RLR']] <- train(
    train_recipe, data = data, method = 'glmnet',
    trControl = train_control, tuneGrid = glmnet_grid,
    metric = "ROC", family = 'binomial')

  message("=== Training Random Forest (ranger) with ", n_folds, " fold CV repeated ",
          n_repeats, " times ===")
  message("Testing ", nrow(rf_grid), " hyperparameter combinations...")
  models[['RF']] <- train(
    train_recipe, data = data, method = 'ranger',
    trControl = train_control, tuneGrid = rf_grid,
    metric = "ROC")

  message("=== Training XGBoost with ", n_folds, " fold CV repeated ",
          n_repeats, " times ===")
  message("Testing ", nrow(xgb_grid), " hyperparameter combinations...")
  models[['XGB']] <- train(
    train_recipe, data = data, method = 'xgbTree',
    trControl = train_control, tuneGrid = xgb_grid,
    metric = "ROC", nthread = 1)

  stopCluster(cl)

  train_objs <- purrr::list_flatten(models)
  saveRDS(train_objs,
          file.path(res_path, paste0('train_objects', output_suffix, '.rds')))

  # ---- Extract per-fold performance ----

  all_perf <- map(train_objs, ~.x$resample) |>
    bind_rows(.id = 'model_id') |>
    mutate(model = sub(".*_", "", model_id),
           rep = sub("_.*", "", model_id),
           .after = model_id) |>
    relocate(Resample, .after = rep)

  write.csv(all_perf,
            file.path(res_path, paste0('perf_all', output_suffix, '.csv')),
            row.names = FALSE)

  metric_cols <- c("ROC", "Sens", "Spec", "Accuracy", "BalancedAccuracy",
                   "F1", "Precision", "PPV", "NPV", "Kappa", "MCC", "PR_AUC")

  hyperparam_cols <- setdiff(colnames(all_perf),
                             c("model_id", "model", "rep", "Resample", metric_cols))

  # Mean performance per model and hyperparameter combination
  mean_perf <- all_perf |>
    group_by(model, across(all_of(hyperparam_cols))) |>
    summarise(across(all_of(metric_cols),
                     list(mean = ~mean(.x, na.rm = TRUE),
                          sd = ~sd(.x, na.rm = TRUE)),
                     .names = "{.col}_{.fn}"),
              .groups = "drop")

  write.csv(mean_perf,
            file.path(res_path, paste0('mean_perf_by_model', output_suffix, '.csv')),
            row.names = FALSE)

  # ---- Hyperparameter tuning plots ----

  en_plot <- mean_perf |>
    filter(model == 'RLR') |>
    ggplot(aes(x = lambda, y = ROC_mean, color = as.factor(alpha))) +
    geom_line() + geom_point() +
    labs(title = "Elastic Net: Mean ROC by Hyperparameters",
         color = "Alpha", y = "Mean ROC-AUC", x = "Lambda") +
    theme_publ_bw() +
    guides(color = guide_legend(nrow = 2))

  print(en_plot)
  ggsave(plot = en_plot, filename = 'RLR_tuning_plot.pdf', path = res_path,
         width = 8.5, height = 7.5, units = 'cm')

  rf_plot <- mean_perf |>
    filter(model == 'RF') |>
    ggplot(aes(x = min.node.size, y = ROC_mean, color = as.factor(mtry))) +
    geom_line() + geom_point() +
    labs(title = "Random Forest: Mean ROC by Hyperparameters",
         x = "Min Node Size", y = "Mean ROC-AUC", color = "mtry") +
    theme_publ_bw() +
    guides(color = guide_legend(nrow = 2))

  print(rf_plot)
  ggsave(plot = rf_plot, filename = 'RF_tuning_plot.pdf', path = res_path,
         width = 8.5, height = 7.5, units = 'cm')

  # For XGBoost, show distribution of each hyperparameter vs ROC
  xgb_plot_data <- mean_perf |>
    filter(model == 'XGB') |>
    dplyr::select(-any_of(c('splitrule'))) |>
    dplyr::select(-where(~all(is.na(.)))) |>
    pivot_longer(cols = any_of(hyperparam_cols),
                 names_to = "hyperparameter", values_to = "value") |>
    mutate(value = as.factor(value))

  xgb_plots <- lapply(unique(xgb_plot_data$hyperparameter), function(param) {
    xgb_plot_data |>
      filter(hyperparameter == param) |>
      ggplot(aes(x = value, y = ROC_mean)) +
      geom_boxplot(fill = 'lightsteelblue', width = 0.5) +
      labs(title = param, y = "Mean ROC-AUC", x = param) +
      theme_publ_bw() +
      geom_pwc(method = "wilcox.test", label.size = 6 / .pt, step.increase = 0.18) +
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.15)))
  })

  xgb_plot <- wrap_plots(xgb_plots)
  print(xgb_plot)
  ggsave(plot = xgb_plot, filename = 'XGB_tuning_plot.pdf', path = res_path,
         width = 10, height = 12, units = 'cm')

  # Show where best hyperparameters sit within the tested ranges
  best_params <- mean_perf |>
    group_by(model) |>
    slice_max(ROC_mean, n = 1, with_ties = TRUE) |>
    ungroup()

  write.csv(best_params,
            file.path(res_path, paste0('best_models_mean_perf', output_suffix, '.csv')),
            row.names = FALSE)

  params <- mean_perf |>
    dplyr::select(model, all_of(hyperparam_cols)) |>
    dplyr::select(-any_of(c('splitrule'))) |>
    pivot_longer(cols = any_of(hyperparam_cols),
                 names_to = "hyperparameter", values_to = "value") |>
    mutate(group = 'all') |>
    bind_rows(
      pivot_longer(
        dplyr::select(best_params, model, all_of(hyperparam_cols),
                      -any_of(c('splitrule'))),
        cols = any_of(hyperparam_cols),
        names_to = "hyperparameter", values_to = "value") |>
        mutate(group = 'best')) |>
    filter(!is.na(value)) |>
    distinct()

  tune_plot <- params |>
    ggplot(aes(x = value, y = group, color = model)) +
    geom_point() + geom_line() +
    facet_wrap(~hyperparameter, scales = "free", nrow = 4) +
    theme_publ_bw()

  print(tune_plot)
  ggsave(plot = tune_plot, filename = 'tuning_plot.pdf', path = res_path,
         width = 10, height = 12, units = 'cm')

  # ---- Extract predictions ----

  best_param_results <- all_perf |>
    semi_join(best_params, by = hyperparam_cols)

  write.csv(best_param_results,
            file.path(res_path, paste0('best_models_all_perf', output_suffix, '.csv')),
            row.names = FALSE)

  all_pred <- map(train_objs, ~.x$pred) |>
    bind_rows(.id = 'model_id') |>
    mutate(model = sub(".*_", "", model_id),
           rep = sub("_.*", "", model_id),
           .after = model_id) |>
    relocate(Resample, .after = rep)

  write.csv(all_pred,
            file.path(res_path, paste0('all_pred', output_suffix, '.csv')),
            row.names = FALSE)

  best_param_preds <- all_pred |>
    semi_join(best_params, by = hyperparam_cols)

  write.csv(best_param_preds,
            file.path(res_path, paste0('best_models_all_pred', output_suffix, '.csv')),
            row.names = FALSE)

  rm(train_objs)

  # ---- Calculate per-fold variable importance ----

  # Set up parallel processing for variable importance calculation
  plan(multisession, workers = n_cores)

  t1 <- Sys.time()
  set.seed(seed) # Ensure reproducibility of variable importance results
  results <- foreach(i = seq_along(folds),
                     .options.future = list(seed = TRUE)) %dofuture% {

    # Extract fold and repeat identifiers from fold names                
    fold_name <- names(folds)[i]
    rep_id <- sub('.*\\.', '', names(folds)[i])
    fold_id <- sub('\\..*', '', names(folds)[i])

    cat("\n=== Calculating variable importance for", rep_id, fold_id, " ===\n")

    # Create training and test sets for the current fold
    train_indices <- folds[[i]]
    train_data <- data[train_indices, ]
    test_indices <- setdiff(seq_len(nrow(data)), train_indices)
    test_data <- data[test_indices, ]

    # Apply the same recipe as during initial training
    fold_recipe <- recipe(train_data) %>%
      update_role(-all_of(response_var), new_role = "predictor") %>%
      update_role(all_of(response_var), new_role = "outcome") %>%
      step_normalize(any_of('Age'))

    # Prep the recipe on the training data and apply to both training and test sets,
    # as caret train() does during model fitting
    prep_recipe <- prep(fold_recipe, training = train_data)
    train_data <- bake(prep_recipe, new_data = train_data)
    test_data <- bake(prep_recipe, new_data = test_data)

    # Define options for caret train
    ctrl <- trainControl(
      seeds = seeds[[i]], # Use the same seed as in initial training
      method = 'none', # no resampling, fit a single model with the best hyperparameters
      classProbs = TRUE,
      allowParallel = FALSE)

    formula <- as.formula(paste(response_var, "~ ."))

    mod <- vector(mode = 'list', length = 3)
    names(mod) <- c('RLR', 'RF', 'XGB')

    # Get the best hyperparameters for this fold
    en_grid <- best_params |>
      filter(model == 'RLR') |>
      dplyr::select(where(~!all(is.na(.))), -model,
                    -contains('mean'), -contains('sd'))

    # Fit a model on the training data for this fold
    mod[['RLR']] <- train(
      formula, 
      data = train_data, 
      method = 'glmnet',
      trControl = ctrl, 
      tuneGrid = en_grid, 
      metric = "ROC")

    rf_grid <- best_params |>
      filter(model == 'RF') |>
      dplyr::select(where(~!all(is.na(.))), -model,
                    -contains('mean'), -contains('sd'))

    mod[['RF']] <- train(
      formula, 
      data = train_data, 
      method = 'ranger',
      trControl = ctrl, 
      tuneGrid = rf_grid, 
      metric = "ROC",
      importance = 'permutation')

    xgb_grid <- best_params |>
      filter(model == 'XGB') |>
      dplyr::select(where(~!all(is.na(.))), -model,
                    -contains('mean'), -contains('sd'))

    mod[['XGB']] <- train(
      formula, 
      data = train_data,
      method = 'xgbTree',
      trControl = ctrl, 
      tuneGrid = xgb_grid, 
      metric = "ROC",
      nthread = 1)

    # Get model-internal variable importance
    model_var_imp_fold <- map(mod, ~varImp(.x, scale = FALSE)) |>
      map(~.x$importance) |>
      map(~rownames_to_column(.x, var = 'Variable')) |>
      bind_rows(.id = 'model')

    # Calculate permutation variable importance on the held-out test fold
    var_imp_fold <- map(mod, function(fit) {
      vip::vi(
        fit,
        method = 'permute',
        target = response_var,
        metric = 'roc_auc',
        feature_names = predictor_vars,
        train = test_data,
        nsim = 50, # 50 permutations per variable
        event_level = 'first',
        pred_wrapper = function(object, newdata) {
          predict.train(object, newdata, type = "prob")[, positive_class]
        })
    }) |>
      bind_rows(.id = 'model')

    list(
      var_imp = var_imp_fold,
      model_var_imp = model_var_imp_fold,
      final_models = mod)
  }

  t2 <- Sys.time()
  cat("\nVariable importance calculation time:",
      round(difftime(t2, t1, units = "mins"), 2), "minutes\n")

  names(results) <- names(folds)
  plan(sequential)

  # Combine and save variable importance

  var_imp_perm <- map_dfr(results, ~.x$var_imp, .id = "resample")

  write.csv(var_imp_perm,
            file.path(res_path, paste0('var_imp_permutation', output_suffix, '.csv')),
            row.names = FALSE)

  var_imp_model <- map_dfr(results, ~.x$model_var_imp, .id = "resample") |>
    dplyr::rename(Importance = Overall) |>
    group_by(model, resample) |>
    mutate(importance_scaled = (Importance - min(Importance)) / # scale to 0-1 within each fold
             (max(Importance) - min(Importance))) |>
    ungroup()

  write.csv(var_imp_model,
            file.path(res_path, paste0('var_imp_model_based', output_suffix, '.csv')),
            row.names = FALSE)

  # Summary of permutation importance across folds
  var_imp_perm_summary <- var_imp_perm |>
    group_by(model, resample) |>
    mutate(Rank_all = dense_rank(desc(Importance)), # importance rank
           Importance_pos = if_else(Importance <= 0, NA_real_, Importance),
           Rank_pos = dense_rank(desc(Importance_pos))) |> # importance rank among positive contributors only
    group_by(model, Variable) |>
    dplyr::summarise(
      n_total = n(),
      n_pos_contr = sum(Importance > 0), # number of folds where variable had positive importance
      freq_pos_contr = n_pos_contr / n_total, # frequency of positive contribution across folds
      median_rank = median(Rank_all), # median rank across folds
      freq_top10 = sum(Rank_pos <= 10, na.rm = TRUE) / n_total, # frequency of being in top 10 among positive contributors 
      mean_importance = mean(Importance, na.rm = TRUE), # mean importance across folds
      sd_importance = sd(Importance, na.rm = TRUE),
      median_importance = median(Importance, na.rm = TRUE),
      .groups = 'keep') |>
    # Calculate 95% confidence intervals for mean importance
    mutate(se = sd_importance / sqrt(n_total),
           ci_lower = mean_importance - qt(0.975, df = n_total - 1) * se,
           ci_upper = mean_importance + qt(0.975, df = n_total - 1) * se) |>
    arrange(model, -mean_importance) |>
    ungroup()

  write.csv(var_imp_perm_summary,
            file.path(res_path, paste0('var_imp_permutation_summary', output_suffix, '.csv')),
            row.names = FALSE)

  # Summary of model-based importance across folds
  var_imp_model_summary <- var_imp_model |>
    group_by(model, resample) |>
    mutate(Rank_all = dense_rank(desc(Importance)),
           Importance_pos = if_else(Importance <= 0, NA_real_, Importance),
           Rank_pos = dense_rank(desc(Importance_pos))) |>
    group_by(model, Variable) |>
    dplyr::summarise(
      n_total = n(),
      n_selected = sum(Importance > 0), # selected if importance > 0 in a given fold
      selection_frequency = n_selected / n_total, # frequency of selection across folds
      median_rank = median(Rank_all),
      freq_top10 = sum(Rank_pos <= 10, na.rm = TRUE) / n_total,
      mean_importance = mean(Importance, na.rm = TRUE),
      sd_importance = sd(Importance, na.rm = TRUE),
      median_importance = median(Importance, na.rm = TRUE),
      .groups = 'drop') |>
    group_by(model) |>
    mutate(
      importance_scaled = (mean_importance - min(mean_importance)) / # scale to 0-1 within each model
        (max(mean_importance) - min(mean_importance)),
      sd_importance_scaled = sd_importance /
        (max(mean_importance) - min(mean_importance))) |>
    arrange(model, -mean_importance) |>
    ungroup()

  write.csv(var_imp_model_summary,
            file.path(res_path, paste0('var_imp_model_based_summary', output_suffix, '.csv')),
            row.names = FALSE)

  # Variables selected (importance > 0) per fold
  used_vars <- filter(var_imp_model, Importance > 0) |>
    group_by(model, resample) |>
    dplyr::summarise(
      variables = list(Variable),
      n_variables = n(),
      .groups = 'drop') |>
    mutate(variables = map_chr(variables, ~paste(.x, collapse = ', ')))

  write.csv(used_vars,
            file.path(res_path, paste0('selected_vars', output_suffix, '.csv')),
            row.names = FALSE)

  # Save fold models
  final_models <- map(results, ~.x$final_models) |>
    purrr::list_flatten()
  saveRDS(final_models,
          file.path(res_path, paste0('final_models', output_suffix, '.rds')))

  # Save results
  res_list <- list(
    best_params = best_params,
    best_param_results = best_param_results,
    best_param_preds = best_param_preds,
    var_imp_perm = var_imp_perm,
    var_imp_perm_summary = var_imp_perm_summary,
    var_imp_model = var_imp_model,
    var_imp_model_summary = var_imp_model_summary,
    used_vars = used_vars)

  saveRDS(res_list,
          file.path(res_path, paste0('all_results', output_suffix, '.rds')))
  
  # Save session info
  writeLines(capture.output(sessionInfo()),
             con = file.path(res_path, paste0('ML_session_info', output_suffix, '.txt')))

  return(res_list)
}

# --------------------------------------------------------------------------- #
# ----------------------- Performance evaluation ---------------------------- 
# --------------------------------------------------------------------------- #

perf_summary <- function(res) {
  # Summarize cross-validated performance metrics for the best models.
  # res: result object from run_ml_analysis()
  # Returns a data frame with per-metric mean, SD, CI, min, and max.

  res$best_param_results |>
    pivot_longer(cols = c(ROC, BalancedAccuracy, Sens, Spec, PPV, NPV),
                 names_to = "metric", values_to = "value") |>
    group_by(model, metric) |>
    summarise(n = n(),
              mean = mean(value, na.rm = TRUE),
              sd = sd(value, na.rm = TRUE),
              median = median(value, na.rm = TRUE),
              min = min(value, na.rm = TRUE),
              max = max(value, na.rm = TRUE),
              .groups = 'keep') |>
    mutate(se = sd / sqrt(n),
           ci_lower = mean - qt(0.975, df = n - 1) * se,
           ci_upper = mean + qt(0.975, df = n - 1) * se) |>
    ungroup()
}

mean_roc <- function(cv_preds, classes) {
  # Calculate mean ROC curve across CV folds by vertical averaging.
  # cv_preds: data frame of CV predictions for a single model
  # classes: character vector of class names; second element is positive class
  # Returns a data frame with columns FPR, TPR, Lower, Upper (95% CI).

  folds <- unique(cv_preds$Resample)
  roc_list <- vector("list", length(folds))
  auc_values <- numeric(length(folds))
  
  pred_col <- classes[2]
  
  # Calculate ROC for each fold
  roc_list <- lapply(folds, function(fold_id) {
    fold_data <- cv_preds[cv_preds$Resample == fold_id, ]
    pROC::roc(fold_data$obs, fold_data[[pred_col]], 
              levels = classes,
              direction = "<")
  })
  
  # Vertical averaging of ROC curves
  # Define common specificity grid
  spec_grid <- seq(0, 1, length.out = 100)
  tpr_matrix <- matrix(NA, nrow = length(spec_grid), ncol = length(roc_list))
  
  for(i in seq_along(roc_list)) {
    
    # Get the ROC curve coordinates
    specs <- roc_list[[i]]$specificities
    sens <- roc_list[[i]]$sensitivities
    
    # Create dataframe
    roc_df <- data.frame(spec = specs, sens = sens)
    
    # Separate middle points (exclude boundaries)
    middle <- roc_df[roc_df$spec > 0 & roc_df$spec < 1, ]
    
    # Handle duplicates in middle by keeping max sensitivity
    if(nrow(middle) > 0) {
      middle <- middle %>%
        group_by(spec) %>%
        summarise(sens = max(sens), .groups = "drop")
    }
    
    # Add theoretical boundaries
    roc_df <- bind_rows(
      data.frame(spec = 0, sens = 1),  # Upper boundary
      middle,
      data.frame(spec = 1, sens = 0)   # Lower boundary
    ) %>%
      arrange(spec)
    
    # Linear interpolation
    tpr_matrix[, i] <- stats::approx(x = roc_df$spec, 
                                     y = roc_df$sens,
                                     xout = spec_grid,
                                     method = "linear",
                                     rule = 2)$y
  }
  
  # Calculate mean and confidence intervals (t-distribution based)
  mean_tpr <- rowMeans(tpr_matrix, na.rm = TRUE)
  sd_tpr <- apply(tpr_matrix, 1, sd, na.rm = TRUE)
  lower_ci <- sapply(1:nrow(tpr_matrix), function(i) {
    n <- sum(!is.na(tpr_matrix[i, ]))
    mean_tpr[i] - qt(0.975, df = n - 1) * (sd_tpr[i] / sqrt(n))
  })
  upper_ci <- sapply(1:nrow(tpr_matrix), function(i) {
    n <- sum(!is.na(tpr_matrix[i, ]))
    mean_tpr[i] + qt(0.975, df = n - 1) * (sd_tpr[i] / sqrt(n))
  })
  
  # Convert specificity to FPR
  fpr_grid <- 1 - spec_grid
  
  roc_avg_df <- data.frame(
    FPR = fpr_grid,
    TPR = mean_tpr,
    Lower = lower_ci,
    Upper = upper_ci
  )
  
  return(roc_avg_df)
}

calc_optimal_cutoff_perf <- function(preds, positive_class) {
  # Compute performance at the Youden-optimal probability threshold for one fold.
  # preds: data frame of predictions for a single model and fold
  # positive_class: name of the positive class (string)
  # Returns a one-row data frame with threshold, Sens, Spec, BalancedAccuracy, PPV, NPV.

  roc_res <- pROC::roc(response = preds$obs,
                       predictor = preds[[positive_class]],
                       levels = c('No_Cancer', positive_class),
                       direction = "<")

  optimal_coords <- pROC::coords(roc_res, x = "best", best.method = "youden",
                                 ret = c("threshold", "sensitivity", "specificity"))

  cutoff <- optimal_coords$threshold[1]

  preds <- preds |>
    mutate(pred_class = ifelse(!!sym(positive_class) >= cutoff,
                               positive_class, 'No_Cancer'))

  cm <- confusionMatrix(
    factor(preds$pred_class, levels = c('No_Cancer', positive_class)),
    factor(preds$obs,        levels = c('No_Cancer', positive_class)),
    positive = positive_class)

  data.frame(
    threshold = cutoff,
    Sens = cm$byClass['Sensitivity'],
    Spec = cm$byClass['Specificity'],
    BalancedAccuracy = cm$byClass['Balanced Accuracy'],
    PPV = cm$byClass['Pos Pred Value'],
    NPV = cm$byClass['Neg Pred Value'],
    row.names = NULL)
}

summarise_optimal_perf <- function(perf_optimal) {
  # Summarize performance at optimal cutoff across fold
  # perf_optimal: data frame with per-fold performance at optimal cutoff,
  #   output from calc_optimal_cutoff_perf()
  # Returns a data frame with mean, SD, CI, min, and max for each metric.
  
  perf_optimal |>
    pivot_longer(cols = c(threshold, ROC, Sens, Spec, BalancedAccuracy, PPV, NPV),
                 names_to = "metric", values_to = "value") |>
    group_by(model, metric) |>
    summarise(n = n(),
              mean = mean(value, na.rm = TRUE),
              sd = sd(value, na.rm = TRUE),
              median = median(value, na.rm = TRUE),
              min = min(value, na.rm = TRUE),
              max = max(value, na.rm = TRUE),
              .groups = 'drop') |>
    mutate(se = sd / sqrt(n),
           ci_lower = mean - qt(0.975, df = n - 1) * se,
           ci_upper = mean + qt(0.975, df = n - 1) * se)
}

# --------------------------------------------------------------------------- #
# ----------------------- Variable importance helpers ----------------------- #
# --------------------------------------------------------------------------- #

create_display_names <- function(variable) {
  # Abbreviate selected long variable names for plots.
  # variable: character vector of variable names
  # Returns a character vector of display names.
  
  case_when(
    variable == "Lump_swelling_or_obstruction_sensation" ~ "Lump/swelling/obstruct. sens.",
    variable == "Pain_persists_worsens_when_breathing" ~ "Pain persists/worsens w. breat.",
    variable == "Felt_tiredness_weakness_or_lack_of_energy_that_came_and_went" ~ "Fatigue that comes and goes",
    variable == "Poorer_condition_lower_fitness"  ~ "Poorer/lower fitness",
    variable == 'Breathing_sound_whistling' ~ 'Breathing sound: whistling',
    variable == 'Changes_hard_to_describe' ~ 'Eating: Changes hard to descr.',
    variable == 'Legs_do_not_support' ~ 'Weakness in the legs',
    variable == 'Felt_worn_out' ~ 'Feeling worn out',
    variable == 'Thickness_in_throat' ~ 'Tightness in throat',
    variable == 'Persisting_pain' ~ 'Persistent pain',
    variable == 'Gasp_for_air' ~ 'Gasping for air',
    variable == 'Pain_radiated_between_shoulder_blades_to_chest' ~ 'Pain btwn shoulder bl. and chest',
    variable == 'Cold_flu_pneumonia_last_2_y' ~ 'Cold/flu/pneumonia past 2 y',
    TRUE ~ str_replace_all(variable, '_', ' ')) |>
    str_replace(' last ', ' past ') |>
    str_replace(' 2 y', ' 2 yrs')
}

rank_all_metrics <- function(var_imp_summary_merged) {
  # Rank each variable by all six importance metrics and compute a median rank.
  # var_imp_summary_merged: merged importance data frame
  # Returns the input data frame with added rank and median_rank_metrics columns.

  var_imp_summary_merged |>
    group_by(model) |>
    mutate(
      rank_selection_frequency = dense_rank(desc(selection_frequency)),
      rank_freq_top10_model = dense_rank(desc(freq_top10_model)),
      rank_importance_scaled_model = dense_rank(desc(importance_scaled_model)),
      rank_freq_pos_contr = dense_rank(desc(freq_pos_contr)),
      rank_freq_top10_perm = dense_rank(desc(freq_top10_perm)),
      rank_importance_scaled_perm = dense_rank(desc(importance_scaled_perm))) |>
    rowwise() |>
    mutate(median_rank_metrics = median(c_across(starts_with('rank_')))) |>
    ungroup()
}
