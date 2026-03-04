# plotting_functions.R
# ggplot2 themes and plotting functions for ML analysis figures

library(ggplot2)
library(patchwork)
library(RColorBrewer)
library(ggh4x)
library(colorspace)
library(ggpubr)
library(tidytext)

colors <- c('#6EACAF', '#FFADC3', '#EE8866')
names(colors) <- c('RF', 'RLR', 'XGB')

# --------------------------------------------------------------------------- #
# --------------------------------- Themes ---------------------------------- 
# --------------------------------------------------------------------------- #

theme_publ <- function() {
  theme_classic(base_size = 7, base_family = 'sans') +
    theme(legend.position = 'top',
          legend.key.size = unit(0.3, 'cm'),
          axis.text = element_text(size = 6),
          legend.text = element_text(size = 6),
          strip.text = element_text(hjust = 0, size = 7),
          strip.background = element_rect(fill = 'grey90', colour = NA))
}

theme_publ_bw <- function() {
  theme_bw(base_size = 7, base_family = 'sans') +
    theme(legend.position = 'top',
          legend.key.size = unit(0.3, 'cm'),
          axis.text = element_text(size = 6),
          legend.text = element_text(size = 6),
          strip.text = element_text(hjust = 0, size = 7))
}

# --------------------------------------------------------------------------- #
# --------------------------- Performance plots ----------------------------- 
# --------------------------------------------------------------------------- #

plot_perf <- function(perf_summary) {
  # Bar plot of model performance metrics at default (0.5) probability threshold.
  # perf_summary: output from perf_summary()
  # Returns a ggplot object.

  perf_summary <- perf_summary |>
    filter(metric != 'ROC') |>
    mutate(metric = factor(
      metric, levels = c("Sens", "Spec", 'BalancedAccuracy', 'PPV', 'NPV'),
      labels = c("Sensitivity", "Specificity", 'Accuracy (Balanced)', 'PPV', 'NPV')))

  ggplot(perf_summary, aes(x = metric, y = mean, fill = model)) +
    geom_bar(stat = "identity", position = "dodge") +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2,
                  position = position_dodge(width = 0.9)) +
    geom_text(aes(y = ci_upper + 0.03, label = round(mean, 2)), size = 6 / .pt,
              position = position_dodge(width = 0.9)) +
    labs(x = "Metric", y = "Mean", fill = 'Model') +
    theme_publ_bw() +
    theme(legend.position = "right",
          axis.text.x = element_text(angle = 20, hjust = 1)) +
    scale_fill_manual(values = colors) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05)), limits = c(0, 1))
}

plot_perf2 <- function(perf_summary) {
  # Bar plot of model performance by predictor set (all, background only, symptoms only).
  # perf_summary: output from perf_summary() with a Variable_Set column
  # Returns a ggplot object.

  perf_summary <- perf_summary |>
    filter(metric %in% c('ROC', 'Sens', 'Spec')) |>
    mutate(metric = factor(
      metric, levels = c("ROC", "Sens", "Spec"),
      labels = c("ROC-AUC", "Sensitivity", "Specificity")))

  ggplot(perf_summary, aes(x = Variable_Set, y = mean, fill = model)) +
    geom_bar(stat = "identity", position = "dodge", show.legend = FALSE) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2,
                  position = position_dodge(width = 0.9)) +
    facet_wrap(~metric, scales = "free_y") +
    labs(title = "Cross validated model performance",
         x = "Model", y = "Mean cross validated performance") +
    theme_publ_bw() +
    scale_fill_manual(values = colors) +
    theme(legend.position = "right") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05)), limits = c(0, 1))
}

# --------------------------------------------------------------------------- #
# -------------------------------- ROC curves ------------------------------- 
# --------------------------------------------------------------------------- #

roc_plot <- function(mean_rocs, perf_summary, classes, title) {
  # Mean ROC curve plot for a single analysis (one positive class).
  # mean_rocs: output from mean_roc() grouped by model
  # perf_summary: output from perf_summary()
  # classes: character vector of class names (positive class second)
  # title: plot title string
  # Returns a ggplot object.

  mean_rocs |>
    ggplot(aes(x = FPR, y = TPR, color = model)) +
    geom_abline(linetype = 'dotted') +
    geom_ribbon(
      aes(ymin = Lower, ymax = Upper, fill = model),
      alpha = 0.2,
      color = NA) +
    geom_path(linewidth = 0.5) +
    coord_equal() +
    theme_publ_bw() +
    scale_y_continuous(limits = c(0, 1)) +
    scale_color_manual(values = colors, aesthetics = c('color', 'fill')) +
    labs(x = "1 - Specificity", y = "Sensitivity",
         color = "Model", fill = "Model", title = title) +
    geom_text(
      data = perf_summary |>
        filter(metric == 'ROC') |>
        mutate(y = c(0.3, 0.23, 0.16), x = 0.52),
      aes(x = x, y = y,
          label = paste0(model, ": ", round(mean, 2), " (CI: ",
                         round(ci_lower, 2), "-", round(ci_upper, 2), ")"),
          color = model),
      hjust = 0, vjust = 0, size = 6 / .pt,
      show.legend = FALSE) +
    theme(legend.position = "right")
}

roc_plot2 <- function(mean_roc_list, perf_summary, classes, title) {
  # Mean ROC curve plot faceted by model, coloured by predictor set.
  # mean_roc_list: named list of mean_roc() outputs (names = predictor sets)
  # perf_summary: output from perf_summary() with a Variable_Set column
  # classes: character vector of class names (positive class second)
  # title: plot title string
  # Returns a ggplot object.

  mean_rocs <- bind_rows(mean_roc_list, .id = 'Variable_Set')

  set_labels <- c(All = 'All', Background = 'Backgr.', Symptoms = 'Sympt.')
  perf_summary$set_label <- set_labels[perf_summary$Variable_Set]

  mean_rocs |>
    ggplot(aes(x = FPR, y = TPR, color = Variable_Set)) +
    geom_abline(linetype = 'dotted') +
    geom_ribbon(
      aes(ymin = Lower, ymax = Upper, fill = Variable_Set),
      alpha = 0.2, color = NA) +
    geom_path(linewidth = 0.5) +
    facet_wrap(~model) +
    coord_equal() +
    theme_publ_bw() +
    scale_y_continuous(limits = c(0, 1)) +
    scale_color_manual(values = brewer.pal(3, 'Dark2')[c(3, 1, 2)],
                       aesthetics = c('color', 'fill')) +
    labs(x = "1 - Specificity", y = "Sensitivity",
         color = "Predictor set", fill = "Predictor set", title = title) +
    geom_text(
      data = perf_summary |>
        filter(metric == 'ROC') |>
        mutate(y = rep(c(0.3, 0.23, 0.16), each = 3), x = 0.4),
      aes(x = x, y = y,
          label = paste0(set_label, ": ", round(mean, 2), " (CI: ",
                         round(ci_lower, 2), "-", round(ci_upper, 2), ")"),
          color = Variable_Set),
      hjust = 0, vjust = 0, size = 6 / .pt,
      show.legend = FALSE) +
    theme(legend.position = "right")
}

# --------------------------------------------------------------------------- #
# ----------------------------- Calibration --------------------------------- 
# --------------------------------------------------------------------------- #

cal_plot <- function(cal_df, cal_stats, title) {
  # Calibration curve plot.
  # cal_df: data frame of flexible calibration curves from val.prob.ci.2()
  # cal_stats: data frame of calibration intercept and slope per model
  # title: plot title string
  # Returns a ggplot object.

  ggplot(cal_df, aes(x = x, y = y)) +
    geom_abline(slope = 1, intercept = 0, color = 'grey') +
    geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = model), alpha = 0.2) +
    geom_line(aes(color = model), linewidth = 0.5) +
    geom_text(data = cal_stats,
              aes(x = 0.05, y = 0.9,
                  label = paste0('Intercept: ', round(Intercept, 2),
                                 '\nSlope: ', round(Slope, 2))),
              hjust = 0, lineheight = 0.8, size = 6 / .pt) +
    geom_rug(aes(color = model), sides = 'b') +
    labs(x = 'Predicted probability', y = 'Observed frequency',
         title = title, color = 'Model', fill = 'Model') +
    theme_publ_bw() +
    facet_wrap(~model) +
    scale_color_manual(values = colors, aesthetics = c('fill', 'color')) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme(legend.position = 'bottom')
}

# --------------------------------------------------------------------------- #
# ------------------------- Decision curve analysis ------------------------- 
# --------------------------------------------------------------------------- #

dca_plot <- function(dca_res, title) {
  # Decision curve analysis plot with standardised net benefit and a secondary
  # cost:benefit ratio x-axis.
  # dca_res: data frame from bind_rows() of rmda decision_curve() derived.data,
  #   with a model column
  # title: plot title string
  # Returns a ggplot object.

  threshold_breaks <- seq(0, 1, 0.2)
  cb_numeric <- ifelse(threshold_breaks == 1, 100,
                       threshold_breaks / (1 - threshold_breaks))
  cb_numeric[threshold_breaks == 0] <- 1 / 100

  cb_labels <- sapply(cb_numeric, function(x) {
    frac <- as.character(MASS::fractions(x))
    if (!grepl("/", frac)) frac <- paste0(frac, ":1")
    gsub("/", ":", frac)
  })

  ggplot(dca_res, aes(x = thresholds, y = sNB, color = model)) +
    geom_line(lwd = 0.5) +
    geom_ribbon(
      data = dca_res |> filter(model %in% c('RLR', 'RF', 'XGB', 'All')),
      aes(ymin = sNB_lower, ymax = sNB_upper, fill = model),
      alpha = 0.2, color = NA) +
    coord_cartesian(xlim = c(0, 1), ylim = c(-0.15, 1)) +
    scale_color_manual(values = setNames(c(colors, 'grey', 'grey50'),
                                         c(names(colors), 'All', 'None')),
                       aesthetics = c('color', 'fill')) +
    scale_x_continuous(
      breaks = threshold_breaks,
      name = "Threshold probability",
      sec.axis = sec_axis(
        trans = ~ .,
        labels = cb_labels,
        breaks = threshold_breaks,
        name = "Cost:benefit ratio")) +
    labs(y = "Standardized net benefit", color = "Model", fill = 'Model',
         title = title) +
    theme_publ_bw() +
    theme(legend.position = 'inside',
          legend.position.inside = c(0.87, 0.7))
}

# --------------------------------------------------------------------------- #
# ------------------------- Variable importance ----------------------------- 
# --------------------------------------------------------------------------- #

plot_imp_heatmap <- function(importance, top_variables, title) {
  # Heatmap of variable importance metrics for top predictors.
  # importance: merged permutation and model-based importance summary
  # top_variables: top variables per model (median ranked)
  # title: overall plot title string
  # Returns a patchwork ggplot object.

  strip_colors <- scales::hue_pal()(2)
  names(strip_colors) <- c("Model-based", "Permutation-based")

  alg_labels <- c(RLR = 'RLR', RF = 'RF', XGB = 'XGB')

  hm_data <- importance |>
    inner_join(top_variables, by = c('model', 'Variable')) |>
    dplyr::select(Variable_name, model, selection_frequency, freq_top10_model,
                  importance_scaled_model, freq_pos_contr, freq_top10_perm,
                  importance_scaled_perm, Order) |>
    pivot_longer(
      cols = -c(Variable_name, model, Order),
      names_to = 'Metric',
      values_to = 'Value') |>
    mutate(
      Importance_method = case_when(
        Metric %in% c('selection_frequency', 'freq_top10_model',
                      'importance_scaled_model') ~ 'Model-based',
        TRUE ~ 'Permutation-based'),
      Metric = factor(
        Metric,
        levels = c('selection_frequency', 'freq_top10_model', 'importance_scaled_model',
                   'freq_pos_contr', 'freq_top10_perm', 'importance_scaled_perm'),
        labels = c('Selection freq.', 'Freq. top 10', 'Mean importance',
                   'Freq. pos. contrib.', 'Freq. top 10', 'Mean importance')),
      Column_order = case_when(
        Metric %in% c('Selection freq.', 'Freq. pos. contrib.') ~ 1,
        Metric %in% c('Freq. top 10') ~ 2,
        Metric %in% c('Mean importance') ~ 3))

  hm_list <- list()

  for (m in c('RLR', 'RF', 'XGB')) {
    hm_subset <- hm_data |> filter(model == m)

    hm_list[[m]] <- ggplot(hm_subset,
                           aes(x = reorder(Metric, Column_order),
                               y = reorder(Variable_name, -Order),
                               fill = Value)) +
      geom_tile(color = 'white') +
      facet_wrap2(~Importance_method, scales = 'free_x', nrow = 1,
                  strip = strip_themed(
                    background_x = elem_list_rect(fill = strip_colors,
                                                  color = rep('white', 2)),
                    text_x = elem_list_text(color = rep('transparent', 2)))) +
      # Transparent dummy layer to generate the colour legend for importance method
      geom_tile(data = data.frame(Importance_method = names(strip_colors),
                                  x = 1, y = 1),
                aes(x = x, y = y, color = Importance_method),
                alpha = 0, inherit.aes = FALSE, linewidth = 0) +
      scale_fill_gradientn(colors = hcl.colors(3, 'Heat 2', rev = TRUE),
                           limits = c(0, 1)) +
      labs(x = 'Importance metric', y = 'Predictor',
           title = alg_labels[[m]],
           fill = 'Importance', color = 'Importance method') +
      theme_publ() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 5),
            legend.position = 'right',
            axis.line = element_blank(),
            axis.ticks = element_blank(),
            plot.title = element_text(size = 7, hjust = 0),
            panel.spacing.x = unit(0, 'cm'),
            strip.text.x = element_text(margin = margin(0.1, 0, 0.1, 0, 'mm')),
            legend.margin = margin(0, 0, 0, 0),
            legend.box.margin = margin(0, 0, 0, 0),
            plot.margin = margin(1, 1, 1, 1)) +
      guides(color = guide_legend(override.aes = list(alpha = 1,
                                                      fill = strip_colors)))
  }

  wrap_plots(hm_list, ncol = length(hm_list), guides = 'collect') +
    plot_layout(axis_titles = 'collect') +
    plot_annotation(
      title = title,
      theme = theme(plot.title = element_text(size = 8, hjust = 0.5)))
}

var_imp_boxplot <- function(var_imp, top_vars, title, type) {
  # Boxplot of variable importance distributions across CV folds.
  # var_imp: data frame of per-fold importance (permutation or model-based)
  # top_vars: top variables per model (median ranked)
  # title: plot title string
  # type: either 'perm' (permutation-based) or 'model' (model-based)
  # Returns a ggplot object.

  xlabel <- case_when(
    type == 'perm'  ~ 'Variable importance (permutation-based, \u0394AUC)',
    type == 'model' ~ 'Variable importance (model-based)')

  importance_var <- if (type == 'perm') {
    sym('Importance')
  } else if (type == 'model') {
    sym('importance_scaled')
  } else {
    stop("Invalid type. Use 'perm' or 'model'")
  }

  var_imp |>
    semi_join(top_vars, by = c('model', 'Variable')) |>
    mutate(model = factor(model, levels = c('RLR', 'RF', 'XGB'))) |>
    ggplot(aes(x = !!importance_var,
               y = reorder_within(Variable_name, !!importance_var, model),
               fill = model)) +
    geom_jitter(aes(color = model), height = 0.2, width = 0, alpha = 0.8) +
    geom_boxplot(alpha = 0.3, outlier.shape = NA) +
    facet_wrap(~model, scales = "free_y") +
    labs(x = xlabel, y = 'Predictor', fill = 'Model', color = 'Model',
         title = title) +
    theme_publ_bw() +
    theme(legend.position = 'bottom') +
    scale_y_reordered() +
    scale_color_manual(values = colors, aesthetics = c('color', 'fill'))
}
