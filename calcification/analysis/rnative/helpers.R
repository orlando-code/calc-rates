library(metafor)
library(ggplot2)
library(dplyr)

# install ggstance if not already installed
if (!requireNamespace("ggstance", quietly = TRUE)) {
    install.packages("ggstance")
}

library(ggstance)


extreme_filtering <- function(
    dat,
    mods_formula = NULL,
    random_structure,
    plot = TRUE,
    percentiles = 1:10, # percentiles to test for exclusion (1% to 10%)
    extreme_limits = NULL # e.g. list(c(-1000, 1000), c(-500, 500)), or a single vector c(-1000, 1000)
    ) {
    # Helper to extract tidy results from a model
    extract_results <- function(model, scenario, exclusion_type = NA, exclusion_value = NA) {
        coefs <- coef(summary(model))
        data.frame(
            Scenario = scenario,
            ExclusionType = exclusion_type,
            ExclusionValue = exclusion_value,
            Term = rownames(coefs),
            est = coefs[, "estimate"],
            ci.lb = coefs[, "ci.lb"],
            ci.ub = coefs[, "ci.ub"],
            QE = model$QE,
            QEp = model$QEp,
            QM = model$QM,
            QMp = model$QMp,
            k = model$k,
            stringsAsFactors = FALSE
        )
    }

    results_list <- list()
    model_info_list <- list()

    # Full model
    print("Fitting full model...")
    res_full <- rma.mv(yi, vi,
        mods = mods_formula,
        random = random_structure,
        data = dat, method = "REML"
    )
    results_list[[length(results_list) + 1]] <- extract_results(res_full, "All data", "None (all data)", NA)
    model_info_list[["All data"]] <- list(
        model = res_full,
        data = dat
    )

    # --- Exclude extremes by percentile (symmetric) ---
    for (p in percentiles) {
        q_low <- quantile(dat$yi, p / 100, na.rm = TRUE)
        q_high <- quantile(dat$yi, 1 - p / 100, na.rm = TRUE)
        dat_trim <- subset(dat, yi >= q_low & yi <= q_high)
        scenario <- sprintf("Trim extremes (%.0f%%)", p)
        print(sprintf("Fitting model with extremes trimmed at %.0f%% (%.2f, %.2f)...", p, q_low, q_high))
        if (nrow(dat_trim) < 5) {
            warning(sprintf("Too few rows after trimming at %.0f%% extremes, skipping.", p))
            next
        }
        res_trim <- tryCatch(
            rma.mv(yi, vi,
                mods = mods_formula,
                random = random_structure,
                data = dat_trim, method = "REML"
            ),
            error = function(e) NULL
        )
        if (!is.null(res_trim)) {
            results_list[[length(results_list) + 1]] <- extract_results(
                res_trim, scenario, "Effect size percentile", p
            )
            model_info_list[[scenario]] <- list(
                model = res_trim,
                data = dat_trim,
                q_low = q_low,
                q_high = q_high
            )
        }
    }

    # --- Exclude by variance percentile (top X%) ---
    for (p in percentiles) {
        v_thresh <- quantile(dat$vi, 1 - p / 100, na.rm = TRUE)
        dat_var <- subset(dat, vi <= v_thresh)
        scenario <- sprintf("High variance (top %.0f%%) excluded", p)
        print(sprintf("Fitting model with high-variance points excluded at %.0f%% (v_thresh=%.2f)...", p, v_thresh))
        if (nrow(dat_var) < 5) {
            warning(sprintf("Too few rows after excluding top %.0f%% variance, skipping.", p))
            next
        }
        res_var <- tryCatch(
            rma.mv(yi, vi,
                mods = mods_formula,
                random = random_structure,
                data = dat_var, method = "REML"
            ),
            error = function(e) NULL
        )
        if (!is.null(res_var)) {
            results_list[[length(results_list) + 1]] <- extract_results(
                res_var, scenario, "Variance percentile", p
            )
            model_info_list[[scenario]] <- list(
                model = res_var,
                data = dat_var,
                v_thresh = v_thresh
            )
        }
    }

    # --- Exclude by user-specified extreme value limits ---
    if (!is.null(extreme_limits)) {
        # Accept either a list of limits or a single vector
        if (is.list(extreme_limits)) {
            limits_list <- extreme_limits
        } else if (is.numeric(extreme_limits) && length(extreme_limits) == 2) {
            limits_list <- list(extreme_limits)
        } else {
            stop("extreme_limits must be a list of length-2 numeric vectors or a single length-2 numeric vector.")
        }
        for (lims in limits_list) {
            lim_low <- lims[1]
            lim_high <- lims[2]
            dat_lim <- subset(dat, yi >= lim_low & yi <= lim_high)
            scenario <- sprintf("User limits (%.2f, %.2f)", lim_low, lim_high)
            print(sprintf("Fitting model with user-specified limits: yi in [%.2f, %.2f]...", lim_low, lim_high))
            if (nrow(dat_lim) < 5) {
                warning(sprintf("Too few rows after applying user limits [%.2f, %.2f], skipping.", lim_low, lim_high))
                next
            }
            res_lim <- tryCatch(
                rma.mv(yi, vi,
                    mods = mods_formula,
                    random = random_structure,
                    data = dat_lim, method = "REML"
                ),
                error = function(e) NULL
            )
            if (!is.null(res_lim)) {
                # Ensure ExclusionValue is always a character for consistency
                exclusion_value_str <- sprintf("%.2f:%.2f", lim_low, lim_high)
                userlimits_str <- sprintf("User limits (%.1f, %.1f)", lim_low, lim_high)
                results_list[[length(results_list) + 1]] <- extract_results(
                    res_lim, scenario, userlimits_str, exclusion_value_str
                )
                model_info_list[[scenario]] <- list(
                    model = res_lim,
                    data = dat_lim,
                    lim_low = lim_low,
                    lim_high = lim_high
                )
            }
        }
    }

    # Combine into one tidy df
    # Ensure all ExclusionValue entries are character to avoid bind_rows() type errors
    results_list <- lapply(results_list, function(df) {
        if ("ExclusionValue" %in% names(df)) {
            df$ExclusionValue <- as.character(df$ExclusionValue)
        }
        df
    })
    sens_results <- bind_rows(results_list)

    # Plot
    if (plot) {
        # Plotting: show all scenarios on the same axis, color by ExclusionType, shape by ExclusionValue
        plot_extreme_filtering(sens_results = sens_results)
    }

    return(list(
        results = sens_results,
        full_model = res_full,
        models = model_info_list
    ))
}







# plot_coefficient_comparison <- function(sens_results,
#                                         title = "Sensitivity to extreme values") {
#     p <- ggplot(
#         sens_results,
#         aes(x = est, y = Term, color = Scenario)
#     ) +
#         geom_point(position = position_dodge(width = 0.6)) +
#         geom_errorbar(aes(xmin = ci.lb, xmax = ci.ub, color = Scenario),
#             position = position_dodge(width = 0.6), width = 0.2
#         ) +
#         geom_vline(xintercept = 0, linetype = "dashed") +
#         facet_wrap(~Group, scales = "free_y", ncol = 1) + # 🔑 split into obvious group panels
#         labs(
#             x = "Coefficient estimate (±95% CI)",
#             y = "Coefficient",
#             title = title
#         ) +
#         theme_minimal(base_size = 14) +
#         theme(
#             strip.background = element_rect(fill = "grey90", colour = NA), # clearer boxes
#             strip.text = element_text(face = "bold"),
#             legend.position = "bottom"
#         )
#     print(p)
# }


loo_analysis <- function(dat, mods_formula = NULL, random_structure,
                         study_var = "doi", level = c("sample", "study")) {
    level <- match.arg(level)
    print(paste("Running leave-one-out analysis at level:", level))

    # Determine loop indices
    if (level == "sample") {
        units <- 1:nrow(dat)
    } else if (level == "study") {
        units <- unique(dat[[study_var]])
    }

    # Progress bar setup
    n_units <- length(units)
    pb <- txtProgressBar(min = 0, max = n_units, style = 3)

    # Run leave-one-out fits
    loo_models <- vector("list", n_units)
    for (i in seq_along(units)) {
        u <- units[i]
        if (level == "sample") {
            dat_i <- dat[-u, ]
        } else {
            dat_i <- dat[dat[[study_var]] != u, ]
        }

        fit <- try(
            rma.mv(yi, vi,
                mods = mods_formula,
                random = random_structure,
                data = dat_i, method = "REML"
            ),
            silent = TRUE
        )
        if (!inherits(fit, "try-error")) {
            loo_models[[i]] <- fit
        } else {
            loo_models[[i]] <- NULL
        }
        setTxtProgressBar(pb, i)
    }
    close(pb)

    # Collect results
    loo_df <- do.call(rbind, lapply(seq_along(loo_models), function(i) {
        fit <- loo_models[[i]]
        if (is.null(fit)) {
            return(NULL)
        }

        coefs <- coef(fit)
        ses <- sqrt(diag(vcov(fit)))

        # Extract QE and QM
        QE <- fit$QE
        QM <- fit$QM

        id_label <- if (level == "sample") units[i] else as.character(units[i])

        df <- data.frame(
            Excluded = id_label,
            Level = level,
            Term = names(coefs),
            est = coefs,
            se = ses,
            ci.lb = coefs - 1.96 * ses,
            ci.ub = coefs + 1.96 * ses,
            stringsAsFactors = FALSE
        )
        # Add QE and QM columns (same for all rows in this fit)
        df$QE <- QE
        df$QM <- QM
        df
    }))

    return(loo_df)
}


unit_filtering <- function(dat,
                           mods_formula = NULL,
                           random_structure,
                           grouping_var = "core_grouping",
                           plot = TRUE) {
    # Helper to extract tidy results
    extract_results <- function(model, scenario, subgroup) {
        coefs <- coef(summary(model))
        data.frame(
            Group = subgroup,
            Scenario = scenario,
            Term = rownames(coefs),
            est = coefs[, "estimate"],
            ci.lb = coefs[, "ci.lb"],
            ci.ub = coefs[, "ci.ub"],
            QE = model$QE,
            QEp = model$QEp,
            QM = model$QM,
            QMp = model$QMp,
            k = model$k,
            stringsAsFactors = FALSE
        )
    }

    results_list <- list()
    model_info_list <- list()

    # --- Full model on all data ---
    print("Fitting full model on all data...")
    res_full <- rma.mv(yi, vi,
        mods = mods_formula,
        random = random_structure,
        data = dat, method = "REML"
    )
    results_list[[length(results_list) + 1]] <- extract_results(res_full, "All data", "All")
    model_info_list[["All_all"]] <- list(model = res_full, data = dat)

    # Loop over subgroups (units or other grouping_var)
    for (g in unique(dat[[grouping_var]])) {
        dat_g <- subset(dat, dat[[grouping_var]] == g)
        if (nrow(dat_g) < 5) next # skip too-small groups

        message("Processing group: ", g)

        # Fit model to this subgroup
        print("Fitting model for group...")
        res_g <- tryCatch(
            rma.mv(yi, vi,
                mods = mods_formula,
                random = random_structure,
                data = dat_g, method = "REML"
            ),
            error = function(e) {
                message(sprintf("Model failed for group %s: %s", g, e$message))
                return(NULL)
            }
        )
        if (!is.null(res_g)) {
            results_list[[length(results_list) + 1]] <- extract_results(res_g, "Group only", g)
            model_info_list[[paste0(g, "_group")]] <- list(model = res_g, data = dat_g)
        }
    }
    # Combine into tidy df
    sens_results <- bind_rows(results_list)
    # Plot
    if (plot) {
        plot_coefficient_comparison(sens_results)
    }
    return(list(
        results = sens_results,
        models = model_info_list
    ))
}




influence_filtering <- function(dat,
                                mods_formula = NULL,
                                random_structure,
                                grouping_var = "core_grouping",
                                plot = TRUE) {
    # Helper to extract tidy results
    extract_results <- function(model, scenario, subgroup) {
        coefs <- coef(summary(model))
        data.frame(
            Group = subgroup,
            Scenario = scenario,
            Term = rownames(coefs),
            est = coefs[, "estimate"],
            ci.lb = coefs[, "ci.lb"],
            ci.ub = coefs[, "ci.ub"],
            QE = model$QE,
            QEp = model$QEp,
            QM = model$QM,
            QMp = model$QMp,
            k = model$k,
            stringsAsFactors = FALSE
        )
    }

    results_list <- list()
    model_info_list <- list()

    # Define scenario labels to use consistently across groups
    cooks_scenario_labels <- c(
        "conservative" = "Cooks conservative",
        "liberal" = "Cooks liberal"
    )

    # Loop over subgroups
    for (g in unique(dat[[grouping_var]])) {
        dat_g <- subset(dat, dat[[grouping_var]] == g)
        if (nrow(dat_g) < 5) next # skip too-small groups

        message("Processing group: ", g)

        # --- Full model ---
        res_full <- rma.mv(yi, vi,
            mods = mods_formula,
            random = random_structure,
            data = dat_g, method = "REML"
        )
        results_list[[length(results_list) + 1]] <- extract_results(res_full, "All data", g)
        model_info_list[[paste0(g, "_all")]] <- list(model = res_full, data = dat_g)

        n_obs <- nrow(dat_g)
        n_coef <- length(coef(res_full))
        # Calculate the two standard Cook's distance thresholds
        calculated_cooks_thresholds <- list(
            "conservative" = 4 / (n_obs - n_coef),
            "liberal" = 2 * sqrt(n_coef / (n_obs - n_coef - 1))
        )


        cooks <- compute_cooks_distance(res_full)
        # Use the calculated thresholds, not the function argument
        for (thr_name in names(calculated_cooks_thresholds)) {
            print(paste("Applying", thr_name, "threshold:", round(calculated_cooks_thresholds[[thr_name]], 4)))
            thr <- calculated_cooks_thresholds[[thr_name]]
            cutoff <- thr
            to_exclude <- which(cooks > cutoff)
            scenario_label <- cooks_scenario_labels[[thr_name]]

            if (length(to_exclude) > 0 & length(to_exclude) < nrow(dat_g)) {
                dat_cook <- dat_g[-to_exclude, ]
                res_cook <- rma.mv(yi, vi,
                    mods = mods_formula,
                    random = random_structure,
                    data = dat_cook, method = "REML"
                )
                results_list[[length(results_list) + 1]] <-
                    extract_results(res_cook, scenario_label, g)
                model_info_list[[paste0(g, "_", scenario_label)]] <- list(
                    model = res_cook,
                    data = dat_cook,
                    excluded = to_exclude,
                    cooks_threshold = thr
                )
            }
        }
    }

    # Combine into tidy df
    sens_results <- bind_rows(results_list)

    # Plot
    if (plot) {
        plot_coefficient_comparison()(sens_results)
    }

    return(list(
        results = sens_results,
        models = model_info_list
    ))
}

clim_filtering <- function(dat,
                           mods_formula = NULL,
                           random_structure,
                           grouping_var = "core_grouping",
                           dt_bounds = c(0, 4),
                           dph_bounds = c(-0.4, 0),
                           plot = TRUE) {
    # Helper to extract tidy results
    extract_results <- function(model, scenario, subgroup) {
        coefs <- coef(summary(model))
        data.frame(
            Group = subgroup,
            Scenario = scenario,
            Term = rownames(coefs),
            est = coefs[, "estimate"],
            ci.lb = coefs[, "ci.lb"],
            ci.ub = coefs[, "ci.ub"],
            QE = model$QE,
            QEp = model$QEp,
            QM = model$QM,
            QMp = model$QMp,
            k = model$k,
            stringsAsFactors = FALSE
        )
    }

    results_list <- list()
    model_info_list <- list()

    # Loop over subgroups
    for (g in unique(dat[[grouping_var]])) {
        dat_g <- subset(dat, dat[[grouping_var]] == g)
        if (nrow(dat_g) < 5) next # skip too-small groups

        message("Processing group: ", g)

        # --- Full model ---
        print("Fitting full model...")
        res_full <- rma.mv(yi, vi,
            mods = mods_formula,
            random = random_structure,
            data = dat_g, method = "REML"
        )
        results_list[[length(results_list) + 1]] <- extract_results(res_full, "All data", g)
        model_info_list[[paste0(g, "_all")]] <- list(model = res_full, data = dat_g)

        # --- Climate filtered model ---
        print("Fitting climate model...")
        dat_clim <- subset(
            dat_g,
            dt >= dt_bounds[1] & dt <= dt_bounds[2] &
                dph >= dph_bounds[1] & dph <= dph_bounds[2]
        )
        if (nrow(dat_clim) < 5) next # skip if too few rows after filtering

        res_clim <- rma.mv(yi, vi,
            mods = mods_formula,
            random = random_structure,
            data = dat_clim, method = "REML"
        )
        results_list[[length(results_list) + 1]] <- extract_results(res_clim, "Climate filtered", g)
        model_info_list[[paste0(g, "_clim_filtered")]] <- list(model = res_clim, data = dat_clim)
    }
    # Combine into tidy df
    sens_results <- bind_rows(results_list)
    # Plot
    if (plot) {
        plot_coefficient_comparison(sens_results)
    }
    return(list(
        results = sens_results,
        models = model_info_list
    ))
}


multicollinearity_diagnostics <- function(mods_formula, dat) {
    # Build the model matrix for the moderators (same terms as mods_formula)
    mod_mat <- model.matrix(as.formula(paste("~", deparse(mods_formula))), data = dat)
    # Remove intercept column when computing pairwise cor and condition number if desired
    # Correlation matrix among numeric columns (excluding intercept if present)
    cm <- cor(mod_mat[, colnames(mod_mat) != "(Intercept)"], use = "pairwise.complete.obs")
    cat("Correlation matrix:\n")
    print(round(cm, 3))

    # Condition number (kappa)
    kappa_val <- kappa(mod_mat)
    cat("Condition number (kappa):", kappa_val, "\n")

    # Compute VIFs via regressing each column on the others using lm (VIF = 1/(1-R^2))
    compute_vif <- function(X) {
        vifs <- numeric(ncol(X))
        names(vifs) <- colnames(X)
        for (j in seq_len(ncol(X))) {
            y <- X[, j]
            Xother <- X[, -j, drop = FALSE]
            if (var(y, na.rm = TRUE) == 0 || ncol(Xother) == 0) {
                vifs[j] <- NA
            } else {
                df_lm <- data.frame(y = y, Xother)
                fm <- lm(y ~ ., data = df_lm)
                R2 <- summary(fm)$r.squared
                vifs[j] <- 1 / (1 - R2)
            }
        }
        return(vifs)
    }

    # Use only numeric predictors (drop intercept)
    X_for_vif <- mod_mat[, colnames(mod_mat) != "(Intercept)", drop = FALSE]
    vifs <- compute_vif(X_for_vif)
    cat("VIFs:\n")
    print(round(vifs, 3))

    invisible(list(correlation = cm, kappa = kappa_val, vif = vifs))
}

#' Compute Cook's Distance for metafor models
#' @param fit A fitted rma.mv object
#' @param parallel Whether to use parallel processing
#' @param ncpus Number of CPU cores to use
#' @return Vector of Cook's distances
compute_cooks_distance <- function(fit, parallel = "multicore", ncpus = 64) {
    cd <- cooks.distance(fit,
        progbar = TRUE,
        reestimate = FALSE,
        parallel = parallel,
        ncpus = ncpus,
        cl = NULL
    )
    return(cd)
}

#' Compute DFBETAS for metafor models
#' @param fit A fitted rma.mv object
#' @param parallel Whether to use parallel processing
#' @param ncpus Number of CPU cores to use
#' @return Matrix of DFBETAS values
compute_dfbetas <- function(fit, parallel = "multicore", ncpus = 64) {
    dfb <- dfbetas(fit,
        progbar = TRUE,
        reestimate = FALSE,
        parallel = parallel,
        ncpus = ncpus,
        cl = NULL
    )
    return(dfb)
}

#' Compute hat values (leverage) for metafor models
#' @param fit A fitted rma.mv object
#' @return Vector of hat values
compute_hatvalues <- function(fit) {
    hat <- hatvalues(fit)
    return(hat)
}

#' Comprehensive influence diagnostics
#' @param fit A fitted rma.mv object
#' @param dat The data frame used to fit the model
#' @param parallel Whether to use parallel processing
#' @param ncpus Number of CPU cores to use
#' @return List containing all influence measures
compute_influence_measures <- function(fit, dat, parallel = "multicore", ncpus = 64) {
    # Cook's distance
    cat("Computing Cook's Distance...\n")
    cd <- compute_cooks_distance(fit, parallel = parallel, ncpus = ncpus)

    # DFBETAS
    # cat("Computing DFBETA...\n")
    # dfb <- compute_dfbetas(fit, parallel = parallel, ncpus = ncpus)

    # Hat values
    cat("Computing Hat values...\n")
    hat <- compute_hatvalues(fit)

    # Create comprehensive data frame
    inf_measures <- data.frame(
        doi = dat$doi,
        hat = hat,
        cooks.distance = cd
        # dfb
    )

    return(inf_measures)
}

#' Identify influential studies using Cook's distance threshold
#' @param inf_measures Data frame of influence measures
#' @param fit The fitted model
#' @param dat The original data frame
#' @param method Method for threshold calculation ("conservative", "liberal", "custom")
#' @param custom_threshold Custom threshold value (if method = "custom")
#' @return List with influential studies and threshold used
identify_influential_studies <- function(inf_measures, fit, dat,
                                         method = "conservative",
                                         custom_threshold = NULL) {
    n_coef <- length(coef(fit))
    n_obs <- nrow(dat)

    # Calculate threshold based on method
    threshold <- switch(method,
        "conservative" = 4 / (n_obs - n_coef),
        "liberal" = 2 * sqrt(n_coef / (n_obs - n_coef - 1)),
        "custom" = custom_threshold
    )

    if (is.null(threshold)) {
        stop(sprintf("Invalid method or missing custom_threshold: %s", method))
    }

    # Identify influential samples
    influential_idx <- inf_measures$cooks.distance > threshold
    influential_samples <- inf_measures[influential_idx, ]

    cat(sprintf("Using %s threshold: %.6f\n", method, threshold))
    cat(sprintf(
        "Found %d influential observations from %d unique studies (containing %d observations total)\n",
        nrow(influential_samples),
        length(unique(influential_samples$doi)),
        sum(dat$doi %in% unique(influential_samples$doi))
    ))

    return(list(
        samples = influential_samples,
        studies = unique(influential_samples$doi),
        threshold = threshold,
        method = method
    ))
}

#' Refit model without influential studies
#' @param dat Original data frame
#' @param influential_result Result from identify_influential_studies()
#' @param mods_formula Model formula
#' @param random_structure Random effects structure
#' @param remove_level Level of removal ("observation" or "study")
#' @return Fitted model without influential studies
refit_without_influential <- function(dat, influential_result,
                                      mods_formula, random_structure, remove_level = "sample") {
    # Remove influential points
    if (remove_level == "study") {
        dat_clean <- subset(dat, !(doi %in% influential_result$unique_dois))
        cat(sprintf("Removed all results corresponding to %d studies (%d samples)\n", length(influential_result$unique_dois), nrow(dat) - nrow(dat_clean)))
    } else if (remove_level == "sample") {
        dat_clean <- dat[!(rownames(dat) %in% rownames(influential_result$samples)), ]
        cat(sprintf("Removed %d influential observation(s)\n", nrow(dat) - nrow(dat_clean)))
    } else {
        stop("Invalid remove_level. Use 'observation' or 'study'.")
    }
    sprintf("Fitting model without influential points (%d samples)...", nrow(dat_clean))
    # Refit model
    fit_clean <- rma.mv(
        yi = yi, V = vi,
        mods = mods_formula,
        random = random_structure,
        data = dat_clean,
        method = "REML"
    )

    return(list(
        fit = fit_clean,
        data = dat_clean,
        removed_studies = influential_result$unique_dois
    ))
}

#' Compare models with and without influential studies
#' @param original_fit Original fitted model
#' @param clean_result Result from refit_without_influential()
#' @return Comparison summary
compare_models <- function(original_fit, clean_result) {
    clean_fit <- clean_result$fit

    # Extract coefficients and standard errors
    orig_coef <- coef(original_fit)
    clean_coef <- coef(clean_fit)

    orig_se <- sqrt(diag(vcov(original_fit)))
    clean_se <- sqrt(diag(vcov(clean_fit)))

    # Create comparison table
    comparison <- data.frame(
        parameter = names(orig_coef),
        original_coef = orig_coef,
        original_se = orig_se,
        clean_coef = clean_coef,
        clean_se = clean_se,
        coef_change = clean_coef - orig_coef,
        se_change = clean_se - orig_se,
        pct_change = ((clean_coef - orig_coef) / orig_coef) * 100
    )

    # Model fit statistics
    model_stats <- data.frame(
        model = c("Original", "Clean"),
        n_studies = c(
            length(unique(original_fit$data$doi)),
            length(unique(clean_result$data$doi))
        ),
        n_obs = c(nrow(original_fit$data), nrow(clean_result$data)),
        tau2 = c(original_fit$tau2, clean_fit$tau2),
        I2 = c(original_fit$I2, clean_fit$I2),
        QE = c(original_fit$QE, clean_fit$QE),
        QEp = c(original_fit$QEp, clean_fit$QEp)
    )

    return(list(
        coefficient_comparison = comparison,
        model_statistics = model_stats,
        removed_studies = clean_result$removed_studies
    ))
}


#' Create a prediction grid for dt and dph
#' @param dat Data frame with original data
#' @param varnames List with names for dt and dph variables
#' @param n_points Number of points in each dimension
#' @return Data frame grid with dt, dph, and their centered/scaled versions
create_prediction_grid <- function(dat, varnames = list(dt = "dt", dph = "dph"), n_points = 50) {
    dt_seq <- seq(min(dat[[varnames$dt]], na.rm = TRUE), max(dat[[varnames$dt]], na.rm = TRUE), length = n_points)
    dph_seq <- seq(min(dat[[varnames$dph]], na.rm = TRUE), max(dat[[varnames$dph]], na.rm = TRUE), length = n_points)
    grid <- expand.grid(dt = dt_seq, dph = dph_seq)

    # Add all possible transformations to handle different model types
    # Centered versions (mean-subtracted)
    grid$dt_c <- grid$dt - mean(dat[[varnames$dt]], na.rm = TRUE)
    grid$dph_c <- grid$dph - mean(dat[[varnames$dph]], na.rm = TRUE)

    # Scaled-only versions (divided by SD, not centered)
    grid$dt_s <- grid$dt / sd(dat[[varnames$dt]], na.rm = TRUE)
    grid$dph_s <- grid$dph / sd(dat[[varnames$dph]], na.rm = TRUE)

    # Standardized versions (centered and scaled)
    grid$dt_z <- (grid$dt - mean(dat[[varnames$dt]], na.rm = TRUE)) / sd(dat[[varnames$dt]], na.rm = TRUE)
    grid$dph_z <- (grid$dph - mean(dat[[varnames$dph]], na.rm = TRUE)) / sd(dat[[varnames$dph]], na.rm = TRUE)

    return(grid)
}


backtransform_coefficients <- function(b, V, dat,
                                       varnames = list(dt = "dt", dph = "dph"),
                                       znames = list(dt = "dt_z", dph = "dph_z"),
                                       snames = list(dt = "dt_s", dph = "dph_s"),
                                       quadnames = list(dt2 = "I(dt^2)", dph2 = "I(dph^2)"),
                                       zquadnames = list(dt2 = "I(dt_z^2)", dph2 = "I(dph_z^2)"),
                                       squadnames = list(dt2 = "I(dt_s^2)", dph2 = "I(dph_s^2)"),
                                       intnames = c("dt:dph", "dt_z:dph_z", "dt_s:dph_s"),
                                       quadintnames = c("I(dt^2):I(dph^2)", "I(dt_z^2):I(dph_z^2)", "I(dt_s^2):I(dph_s^2)")) {
    # compute means and SDs
    mean_dt <- mean(dat[[varnames$dt]], na.rm = TRUE)
    mean_dph <- mean(dat[[varnames$dph]], na.rm = TRUE)
    sd_dt <- sd(dat[[varnames$dt]], na.rm = TRUE)
    sd_dph <- sd(dat[[varnames$dph]], na.rm = TRUE)

    b_unit <- b
    scales <- rep(1, length(b))
    names(scales) <- names(b)

    # --- Helper: rescale coefficients and compute scale factors ---
    rescale <- function(terms, scale_fun) {
        existing <- intersect(terms, names(b))
        if (length(existing) > 0) {
            b_unit[existing] <<- b[existing] / scale_fun(existing)
            scales[existing] <<- 1 / scale_fun(existing)
        }
    }

    # Linear
    rescale(c(znames$dt, snames$dt), function(.) rep(sd_dt, length(.)))
    rescale(c(znames$dph, snames$dph), function(.) rep(sd_dph, length(.)))

    # Quadratic
    rescale(c(zquadnames$dt2, squadnames$dt2), function(.) rep(sd_dt^2, length(.)))
    rescale(c(zquadnames$dph2, squadnames$dph2), function(.) rep(sd_dph^2, length(.)))

    # Interactions - determine scaling based on variable types
    interaction_scales <- sapply(intnames, function(nm) {
        if (nm %in% names(b)) {
            if (nm == "dt_z:dph_z" || nm == "dt_s:dph_s") {
                # Both variables are scaled/standardized
                sd_dt * sd_dph
            } else if (nm == "dt:dph") {
                # Both variables are raw - no scaling needed
                1
            } else {
                # Mixed case - would need more specific logic
                warning(paste("Unknown interaction pattern:", nm))
                1
            }
        } else {
            1
        }
    })

    for (i in seq_along(intnames)) {
        nm <- intnames[i]
        if (nm %in% names(b)) {
            scale_factor <- interaction_scales[i]
            b_unit[nm] <- b[nm] / scale_factor
            scales[nm] <- 1 / scale_factor
        }
    }

    # Quadratic interactions - similar logic
    quadint_scales <- sapply(quadintnames, function(nm) {
        if (nm %in% names(b)) {
            if (nm == "I(dt_z^2):I(dph_z^2)" || nm == "I(dt_s^2):I(dph_s^2)") {
                # Both variables are scaled/standardized
                sd_dt^2 * sd_dph^2
            } else if (nm == "I(dt^2):I(dph^2)") {
                # Both variables are raw - no scaling needed
                1
            } else {
                # Mixed case
                warning(paste("Unknown quadratic interaction pattern:", nm))
                1
            }
        } else {
            1
        }
    })

    for (i in seq_along(quadintnames)) {
        nm <- quadintnames[i]
        if (nm %in% names(b)) {
            scale_factor <- quadint_scales[i]
            b_unit[nm] <- b[nm] / scale_factor
            scales[nm] <- 1 / scale_factor
        }
    }

    # --- Intercept adjustment ---
    # Only adjust intercept if standardized (z) variables are present
    # (scaled-only variables don't require intercept adjustment)
    if ("intrcpt" %in% names(b)) {
        # Only consider standardized (z) variables for intercept adjustment
        z_terms <- c(znames$dt, znames$dph, zquadnames$dt2, zquadnames$dph2)
        z_intnames <- intnames[grepl("_z:", intnames)]
        z_quadintnames <- quadintnames[grepl("_z.*_z", quadintnames)]

        all_z_terms <- c(z_terms, z_intnames, z_quadintnames)
        existing_z_terms <- intersect(all_z_terms, names(b))

        if (length(existing_z_terms) > 0) {
            intercept_shift <- sum(sapply(existing_z_terms, function(nm) {
                if (nm == znames$dt) {
                    # For dt_z: coefficient * (mean_dt / sd_dt)
                    b[nm] * (mean_dt / sd_dt)
                } else if (nm == znames$dph) {
                    # For dph_z: coefficient * (mean_dph / sd_dph)
                    b[nm] * (mean_dph / sd_dph)
                } else if (nm == zquadnames$dt2) {
                    # For I(dt_z^2): coefficient * (mean_dt / sd_dt)^2
                    b[nm] * (mean_dt / sd_dt)^2
                } else if (nm == zquadnames$dph2) {
                    # For I(dph_z^2): coefficient * (mean_dph / sd_dph)^2
                    b[nm] * (mean_dph / sd_dph)^2
                } else if (nm %in% z_intnames) {
                    # For dt_z:dph_z: coefficient * (mean_dt / sd_dt) * (mean_dph / sd_dph)
                    b[nm] * (mean_dt / sd_dt) * (mean_dph / sd_dph)
                } else if (nm %in% z_quadintnames) {
                    # For I(dt_z^2):I(dph_z^2): coefficient * (mean_dt / sd_dt)^2 * (mean_dph / sd_dph)^2
                    b[nm] * (mean_dt / sd_dt)^2 * (mean_dph / sd_dph)^2
                } else {
                    0
                }
            }))
            b_unit["intrcpt"] <- b["intrcpt"] - intercept_shift
        }
    }

    # --- Transform variance-covariance matrix ---
    V_unit <- diag(scales) %*% V %*% diag(scales)

    return(list(b_unit = b_unit, V_unit = V_unit))
}


#' Predict surface values on a grid, with optional backtransformation and plotting
#' @param fit Fitted model
#' @param dat Data frame with original data
#' @param mods_formula Model formula for predictions
#' @param varnames,znames,snames,zquadnames,intname See above
#' @param backtransform Logical, whether to backtransform coefficients
#' @param plot Logical, whether to plot the surface
#' @param plot_anomalies Logical, whether to plot anomalies relative to control (dt=0, dph=0)
#' @param fig_width,fig_height Figure dimensions for plot
#' @return Data frame grid with predictions (invisible)
predict_surface <- function(fit, dat, mods_formula,
                            varnames = list(dt = "dt", dph = "dph"),
                            znames = list(dt = "dt_z", dph = "dph_z"),
                            snames = list(dt = "dt_s", dph = "dph_s"),
                            quadnames = list(dt2 = "I(dt^2)", dph2 = "I(dph^2)"),
                            zquadnames = list(dt2 = "I(dt_z^2)", dph2 = "I(dph_z^2)"),
                            squadnames = list(dt2 = "I(dt_s^2)", dph2 = "I(dph_s^2)"),
                            intnames = c("dt:dph", "dt_z:dph_z", "dt_s:dph_s"),
                            quadintnames = c("I(dt^2):I(dph^2)", "I(dt_z^2):I(dph_z^2)", "I(dt_s^2):I(dph_s^2)"),
                            backtransform = TRUE, plot = TRUE, plot_anomalies = FALSE,
                            fig_width = 6, fig_height = 6, fill_limits = NULL, title = NULL) {
    b <- coef(fit)
    V <- vcov(fit)
    grid <- create_prediction_grid(dat, varnames = varnames, n_points = 50)

    if (backtransform) {
        bt <- backtransform_coefficients(b, V, dat, varnames = varnames, znames = znames, snames = snames, quadnames = quadnames, zquadnames = zquadnames, squadnames = squadnames, intnames = intnames, quadintnames = quadintnames)
        b_unit <- bt$b_unit
        V_unit <- bt$V_unit
    } else {
        b_unit <- b
        V_unit <- V
    }

    # Drop NAs
    b_unit <- b_unit[!is.na(b_unit)]

    # Build design matrix on raw grid
    Xpred <- model.matrix(as.formula(paste("~", deparse(mods_formula))), data = grid)

    # Predicted values
    pred <- as.vector(Xpred %*% b_unit)
    pred_se <- sqrt(diag(Xpred %*% V_unit %*% t(Xpred)))
    # pred_lo <- pred - 1.96 * pred_se
    # pred_hi <- pred + 1.96 * pred_se

    grid$pred <- pred
    grid$pred_se <- pred_se

    if (plot) {
        plot_surface_contour(grid, pred, title = title, fig_width = fig_width, fig_height = fig_height, plot_anomalies = plot_anomalies, fill_limits = fill_limits)
    }
    invisible(grid)
    return(list(grid = grid, pred = pred))
}


# PLOTTING

plot_loo <- function(loo_df, full_model = NULL) {
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(patchwork)

    # Get the unique terms (up to 4)
    terms <- unique(loo_df$Term)
    if (length(terms) > 4) {
        warning("More than 4 terms found; only the first 4 will be plotted.")
        terms <- terms[1:4]
    }

    # Optionally add reference lines from full model
    ref_lines <- NULL
    if (!is.null(full_model)) {
        coefs <- coef(full_model)
        ci <- confint(full_model)
        ref_lines <- data.frame(
            Term = names(coefs),
            est = coefs,
            ci.lb = ci$random["ci.lb", ],
            ci.ub = ci$random["ci.ub", ]
        )
    }

    # Only keep rows for the selected terms
    loo_df <- loo_df[loo_df$Term %in% terms, ]

    # To ensure a common y axis, get the full set of Excluded labels in the order of mean(est) across all terms
    # This ensures all plots share the same y axis and order
    excluded_levels <- loo_df %>%
        group_by(Excluded) %>%
        summarize(mean_est = mean(est, na.rm = TRUE)) %>%
        arrange(mean_est) %>%
        pull(Excluded)

    # Set Excluded as a factor with the same levels for all terms
    loo_df$Excluded <- factor(loo_df$Excluded, levels = excluded_levels)

    # Add a Term factor for faceting
    loo_df$Term <- factor(loo_df$Term, levels = terms)

    # Compute median estimate for each term
    median_df <- loo_df %>%
        group_by(Term) %>%
        summarize(median_est = median(est, na.rm = TRUE))

    # --- Main coefficient panel ---
    p_coef <- ggplot(loo_df, aes(y = Excluded, x = est)) +
        geom_pointrange(aes(xmin = ci.lb, xmax = ci.ub), color = "steelblue") +
        facet_grid(. ~ Term, scales = "free_x", switch = "x") +
        labs(
            x = "Estimate (95% CI)",
            y = ifelse(length(unique(loo_df$Level)) == 1 && unique(loo_df$Level) == "row",
                "Left-out row", "Left-out study"
            ),
            title = "Leave-one-out sensitivity analysis"
        ) +
        theme_bw(base_size = 12) +
        theme(
            axis.text.y = element_text(size = 8),
            strip.placement = "outside",
            strip.background = element_blank(),
            panel.spacing = unit(0.5, "lines")
        ) +
        # Add vertical dotted line for median estimate per term
        geom_vline(
            data = median_df,
            aes(xintercept = median_est, linetype = "Median"),
            color = "grey40"
        ) +
        scale_linetype_manual(
            name = "Reference",
            values = c("Median" = "dashed"),
            labels = c("Median estimate"),
        ) +
        theme(
            legend.position = "bottom"
        ) +
        guides(
            linetype = guide_legend(order = 1)
        )

    # Add reference lines if available
    if (!is.null(ref_lines)) {
        for (i in seq_along(terms)) {
            term <- terms[i]
            if (term %in% ref_lines$Term) {
                ref <- ref_lines[ref_lines$Term == term, ]
                p_coef <- p_coef + geom_vline(
                    xintercept = ref$est,
                    linetype = "dashed", color = "red",
                    data = data.frame(Term = factor(term, levels = terms))
                )
            }
        }
    }

    # --- QE and QM panels ---
    # Prepare data for QE and QM
    # Use the same Excluded factor levels for y axis
    qe_df <- loo_df %>%
        select(Excluded, QE) %>%
        distinct()
    qe_median_est <- median(loo_df$QE, na.rm = TRUE)
    qm_df <- loo_df %>%
        select(Excluded, QM) %>%
        distinct()
    qm_df$QM <- qm_df$QM
    qm_median_est <- median(loo_df$QM, na.rm = TRUE)

    # QE panel
    p_qe <- ggplot(qe_df, aes(y = Excluded, x = QE)) +
        geom_point(color = "darkorange", size = 1.5) +
        labs(x = "QE", y = NULL, title = "QE") +
        theme_bw(base_size = 12) +
        geom_vline(
            data = qe_df,
            aes(xintercept = qe_median_est),
            color = "grey40", linetype = "dashed"
        ) +
        theme(
            axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.title.y = element_blank(),
            plot.title = element_text(size = 11, hjust = 0.5),
            panel.grid.major.y = element_blank(),
            panel.grid.minor.y = element_blank(),
            panel.spacing = unit(0.5, "lines")
        )

    # QM panel
    p_qm <- ggplot(qm_df, aes(y = Excluded, x = QM)) +
        geom_point(color = "darkgreen", size = 1.5) +
        labs(x = "QM", y = NULL, title = "QM") +
        theme_bw(base_size = 12) +
        geom_vline(
            data = qm_df,
            linetype = "dashed",
            aes(xintercept = qm_median_est),
            color = "grey40"
        ) +
        theme(
            axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.title.y = element_blank(),
            plot.title = element_text(size = 11, hjust = 0.5),
            panel.grid.major.y = element_blank(),
            panel.grid.minor.y = element_blank(),
            panel.spacing = unit(0.5, "lines")
        )

    # Remove y axis text from QE and QM, keep only on main panel
    # Compose the panels using patchwork
    # p_coef | p_qe | p_qm
    p_combined <- p_coef + p_qe + p_qm +
        plot_layout(ncol = 3, widths = c(length(terms), 1, 1))

    return(p_combined)
}


plot_coefficient_comparison <- function(sens_results, title = "Sensitivity to extreme values") {
    p <- ggplot(
        sens_results,
        aes(x = est, y = Term, color = Scenario)
    ) +
        geom_point(position = position_dodge(width = 0.6)) +
        geom_errorbar(aes(xmin = ci.lb, xmax = ci.ub),
            position = position_dodge(width = 0.6), width = 0.2
        ) +
        facet_wrap(~Scenario, ncol = 1, scales = "free_y") +
        geom_vline(xintercept = 0, linetype = "dashed") +
        labs(
            x = "Coefficient estimate (±95% CI)", y = "Coefficient",
            title = title
        ) +
        theme_minimal(base_size = 14) +
        theme(legend.position = "none")
    print(p)
}

plot_cooks_distance <- function(influence_df, model, fig_height = 8, fig_width = 12, main = NULL, file = NULL) {
    # If a file is specified, use a file device (recommended for controlling shape)
    if (!is.null(file)) {
        # Use png device for demonstration; change to pdf() or other as needed
        png(filename = file, width = fig_width, height = fig_height, units = "in", res = 300)
        on.exit(dev.off(), add = TRUE)
    } else {
        # If running interactively, try to open a new device with specified size
        # Note: dev.new() may not work as expected in RStudio or some GUIs
        if (dev.cur() == 1) { # 1 means "null device"
            dev.new(width = fig_width, height = fig_height)
            on.exit(dev.off(), add = TRUE)
        }
    }

    cooks <- influence_df$cooks.distance
    n_obs <- length(cooks)
    n_coef <- length(coef(model))

    thresholds <- list(
        "conservative" = 4 / (n_obs - n_coef),
        "liberal" = 2 * sqrt(n_coef / (n_obs - n_coef - 1))
    )

    # Identify points above thresholds
    above_liberal <- which(cooks > thresholds$liberal)
    above_conservative <- which(cooks > thresholds$conservative)

    # Plot all points in blue
    plot(
        cooks,
        type = "p",
        log = "y", pch = 19, col = "blue",
        main = main,
        xlab = "Observation",
        ylab = "Cook's Distance"
    )

    # Add liberal threshold line
    abline(h = thresholds$liberal, col = "red", lty = 2, lwd = 2)
    # Add conservative threshold line
    abline(h = thresholds$conservative, col = "orange", lty = 2, lwd = 2)

    # Optionally, overlay points above conservative threshold in a different color (e.g., darkred)
    if (length(above_conservative) > 0) {
        points(above_conservative, cooks[above_conservative], col = "red", pch = 19)
    }
    # Overlay points above liberal threshold in red
    if (length(above_liberal) > 0) {
        points(above_liberal, cooks[above_liberal], col = "darkred", pch = 19)
    }
    # Move the legend box to the right, minimizing whitespace
    # legend(
    #     x = "bottomright",
    #     inset = c(-0.0, 0), # move legend box just outside plot area to the right
    #     legend = c(
    #         "All points",
    #         expression("Above liberal: " * frac(4, n[obs] - n[coef])),
    #         expression("Above conservative: " * 2 * sqrt(frac(n[coef], n[obs] - n[coef] - 1)))
    #     ),
    #     col = c("blue", "darkred", "red"), pch = 19,
    #     xpd = TRUE, # allow legend to be outside plot region
    #     y.intersp = 1.2,
    #     x.intersp = 2,
    #     text.col = "black",
    #     horiz = FALSE,
    #     bty = "o"
    # )
}

#' Plot contour plot of predicted surface (dph vs dt)
#' @param grid Data frame with prediction grid and predictions
#' @param pred Vector of predicted values
#' @param fig_width Figure width in inches
#' @param fig_height Figure height in inches
#' @param plot_anomalies Logical, whether to plot anomalies relative to control (dt=0, dph=0)
#' @return ggplot object (invisible)
plot_surface_contour <- function(grid, pred, title = NULL, fig_width = 6, fig_height = 6, plot_anomalies = FALSE, fill_limits = NULL) {
    library(ggplot2)

    # Calculate anomalies if requested
    if (plot_anomalies) {
        # Find the prediction closest to control condition (dt=0, dph=0)
        control_idx <- which.min(abs(grid$dt - 0) + abs(grid$dph - 0))
        control_prediction <- pred[control_idx]

        cat("Control prediction (at dt=0, dph=0):", round(control_prediction, 4), "\n")
        cat("Subtracting control value to create anomalies\n")

        # Calculate anomalies (deviations from control)
        pred_plot <- pred - control_prediction
        plot_title_suffix <- " (Anomalies from Control)"
        legend_title <- "Calcification anomaly relative to control (%)"

        # Add anomaly column to grid for contour lines
        grid$pred_anomaly <- pred_plot
        z_var <- pred_plot
    } else {
        pred_plot <- pred
        plot_title_suffix <- ""
        legend_title <- "Predicted percentage relative calcification rate"
        z_var <- pred_plot
    }

    # Add the plotting variable to the grid
    grid$pred_plot <- pred_plot

    # Set up fill scale with optional limits
    fill_scale <- scale_fill_gradient2(
        low = "red",
        mid = "white",
        high = "blue",
        midpoint = 0,
        name = legend_title
    )
    if (!is.null(fill_limits)) {
        fill_scale <- fill_scale$clone()
        fill_scale$limits <- fill_limits
    }

    # Compose the plot title
    if (!is.null(title)) {
        plot_title <- paste0("Predicted Response Surface", plot_title_suffix)
    } else {
        plot_title <- NULL
    }

    p <- ggplot(grid, aes(x = dt, y = dph)) +
        geom_raster(aes(fill = pred_plot), interpolate = TRUE) +
        fill_scale +
        geom_contour(aes(z = z_var), color = "black", alpha = 0.5) +
        labs(
            x = expression(Delta ~ "Temperature"),
            y = expression(Delta ~ "pH"),
            title = plot_title
        ) +
        theme_minimal() +
        theme(
            axis.title.x = element_text(family = "serif", face = "italic", size = 24),
            axis.title.y = element_text(family = "serif", face = "italic", size = 24),
            axis.text.x = element_text(size = 18),
            axis.text.y = element_text(size = 18),
            aspect.ratio = 1,
            legend.position = "top",
            legend.direction = "horizontal",
            legend.title = element_text(size = 16),
            legend.text = element_text(size = 14),
            plot.margin = margin(20, 20, 20, 20, "pt"),
            panel.spacing = unit(0, "pt")
        ) +
        guides(
            fill = guide_colorbar(
                title.position = "top",
                title.hjust = 0.5,
                barwidth = unit(10, "cm"),
                barheight = unit(0.5, "cm"),
                direction = "horizontal"
            )
        ) +
        coord_fixed(expand = FALSE)
    print(p)
    invisible(p)
}


#' @param sens_results Data frame with sensitivity results
#' @param title Plot title
plot_extreme_filtering <- function(sens_results, title = "Sensitivity to extreme values") {
    # Create manual vertical offsets for each ExclusionType and ExclusionValue combination

    type_spacing <- 0.25 # Spacing between different ExclusionTypes
    value_spacing <- 0.02 # Spacing between different ExclusionValues within the same type

    # Convert Term to numeric for positioning
    sens_results$Term_num <- as.numeric(as.factor(sens_results$Term))

    # Create offset groups: first by ExclusionType, then by ExclusionValue within type
    sens_results$.group_key <- paste(sens_results$ExclusionType, sens_results$ExclusionValue, sep = "_")

    # Calculate vertical offsets
    sens_results$y_offset <- sens_results$Term_num

    for (term in unique(sens_results$Term)) {
        term_data <- sens_results[sens_results$Term == term, ]
        term_num <- unique(term_data$Term_num)

        # Get unique ExclusionTypes for this term
        exclusion_types <- unique(term_data$ExclusionType)
        n_types <- length(exclusion_types)

        # Center the ExclusionTypes around the base position
        type_positions <- seq(-type_spacing * (n_types - 1) / 2,
            type_spacing * (n_types - 1) / 2,
            length.out = n_types
        )
        names(type_positions) <- exclusion_types

        # For each ExclusionType, offset the ExclusionValues
        for (etype in exclusion_types) {
            type_data <- term_data[term_data$ExclusionType == etype, ]
            exclusion_values <- unique(type_data$ExclusionValue)
            n_values <- length(exclusion_values)

            if (n_values > 1) {
                # Create offsets for different exclusion values
                value_positions <- seq(-value_spacing * (n_values - 1) / 2,
                    value_spacing * (n_values - 1) / 2,
                    length.out = n_values
                )
                names(value_positions) <- as.character(exclusion_values)

                # Apply offsets
                for (i in 1:nrow(type_data)) {
                    row_idx <- which(sens_results$Term == term &
                        sens_results$ExclusionType == etype &
                        sens_results$ExclusionValue == type_data$ExclusionValue[i])

                    sens_results$y_offset[row_idx] <- term_num +
                        type_positions[etype] +
                        value_positions[as.character(type_data$ExclusionValue[i])]
                }
            } else {
                # Single value, just use type position
                row_idx <- which(sens_results$Term == term & sens_results$ExclusionType == etype)
                sens_results$y_offset[row_idx] <- term_num + type_positions[etype]
            }
        }
    }

    p <- ggplot(
        sens_results,
        aes(
            x = est,
            y = y_offset,
            shape = ExclusionType,
            color = as.numeric(ExclusionValue)
        )
    ) +
        geom_point(size = 2) +
        geom_errorbar(
            aes(xmin = ci.lb, xmax = ci.ub),
            width = 0.02
        ) +
        geom_vline(xintercept = 0, linetype = "dashed") +
        scale_color_viridis_c(
            option = "plasma",
            name = "Exclusion Level",
            na.value = "grey50"
        ) +
        scale_y_continuous(
            breaks = unique(sens_results$Term_num),
            labels = unique(sens_results$Term),
            name = "Coefficient"
        ) +
        labs(
            x = "Coefficient estimate (±95% CI)",
            title = title,
            shape = "Exclusion Type"
        ) +
        theme_minimal(base_size = 14) +
        theme(
            panel.grid.minor.y = element_blank()
        )
    print(p)
}

# plot_old_coefficient_comparison <- function(sens_results,
#                                             title = "Sensitivity to extreme values") {
#     p <- ggplot(
#         sens_results,
#         aes(x = est, y = Term, color = Scenario)
#     ) +
#         geom_point(position = position_dodge(width = 0.6)) +
#         geom_errorbar(aes(xmin = ci.lb, xmax = ci.ub, color = Scenario),
#             position = position_dodge(width = 0.6), width = 0.2
#         ) +
#         geom_vline(xintercept = 0, linetype = "dashed") +
#         facet_wrap(~Group, scales = "free_y", ncol = 1) + # 🔑 split into obvious group panels
#         labs(
#             x = "Coefficient estimate (±95% CI)",
#             y = "Coefficient",
#             title = title
#         ) +
#         theme_minimal(base_size = 14) +
#         theme(
#             strip.background = element_rect(fill = "grey90", colour = NA), # clearer boxes
#             strip.text = element_text(face = "bold"),
#             legend.position = "bottom"
#         )
#     print(p)
# }
