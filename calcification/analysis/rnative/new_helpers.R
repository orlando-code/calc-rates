library(metafor)
library(ggplot2)
library(dplyr)

# install ggstance if not already installed
if (!requireNamespace("ggstance", quietly = TRUE)) {
    install.packages("ggstance")
}

library(ggstance)

# SHARED HELPER FUNCTIONS
# ======================

#' Detect if coefficient terms are scaled based on their names
#' @param term_names Character vector of coefficient term names
#' @return Logical vector indicating which terms are scaled
detect_scaled_terms <- function(term_names) {
    # Patterns that indicate scaled/standardized variables
    scaled_patterns <- c(
        "_z", # standardized (z-scored)
        "_s" # scaled only (divided by SD)
    )

    # Check if any term contains scaling indicators
    sapply(term_names, function(term) {
        any(sapply(scaled_patterns, function(pattern) grepl(pattern, term)))
    })
}

#' Remove scaling indicators from coefficient names
#' @param term_names Character vector of coefficient term names
#' @return Character vector with scaling indicators (_z, _s) removed
#' @examples
#' unscale_coefficient_names(c("dt_z", "dph_s", "I(dt_z^2)", "dt_z:dph_z"))
unscale_coefficient_names <- function(term_names) {
    # Remove _z and _s indicators from variable names
    gsub("_z|_s", "", term_names)
}

#' Standard result extraction from fitted models
#' @param model Fitted rma.mv model
#' @param scenario Scenario description
#' @param exclusion_type Type of exclusion applied
#' @param exclusion_value Value/level of exclusion
#' @param group Optional group identifier
#' @param dat Original data used to fit the model (needed for scaling info)
#' @param store_scaling Whether to store scaling information for later use
extract_results <- function(model, scenario, exclusion_type = NA, exclusion_value = NA, group = NA, dat = NULL, store_scaling = FALSE) {
    coefs <- coef(summary(model))

    # Create base result dataframe
    result <- data.frame(
        Scenario = scenario,
        ExclusionType = exclusion_type,
        ExclusionValue = exclusion_value,
        Group = group,
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

    # Add scaling information if requested and data is available
    if (store_scaling && !is.null(dat)) {
        # Calculate scaling factors for each variable
        varnames <- list(dt = "dt", dph = "dph")
        if ("delta_t" %in% colnames(dat)) varnames$dt <- "delta_t"
        if ("delta_ph" %in% colnames(dat)) varnames$dph <- "delta_ph"

        sd_dt <- if (varnames$dt %in% colnames(dat)) sd(dat[[varnames$dt]], na.rm = TRUE) else NA
        sd_dph <- if (varnames$dph %in% colnames(dat)) sd(dat[[varnames$dph]], na.rm = TRUE) else NA

        # Add scaling columns
        result$sd_dt <- sd_dt
        result$sd_dph <- sd_dph
        result$scaling_available <- !is.na(sd_dt) && !is.na(sd_dph)
    } else {
        # Add empty scaling columns for consistency
        result$sd_dt <- NA
        result$sd_dph <- NA
        result$scaling_available <- FALSE
    }

    return(result)
}

#' Safe model fitting with error handling
#' @param dat Data frame
#' @param mods_formula Model formula
#' @param random_structure Random effects structure
#' @param scenario_name Name for logging
safe_fit_model <- function(dat, mods_formula, random_structure, scenario_name = "model") {
    if (nrow(dat) < 5) {
        warning(sprintf("Too few rows (%d) for %s, skipping.", nrow(dat), scenario_name))
        return(NULL)
    }

    tryCatch(
        rma.mv(yi, vi,
            mods = mods_formula,
            random = random_structure,
            data = dat, method = "REML"
        ),
        error = function(e) {
            message(sprintf("Model failed for %s: %s", scenario_name, e$message))
            return(NULL)
        }
    )
}

#' Generic sensitivity analysis framework
#' @param dat Data frame
#' @param mods_formula Model formula
#' @param random_structure Random effects structure
#' @param filters List of filter functions to apply
#' @param grouping_var Optional grouping variable
#' @param plot Whether to plot results
#' @param plot_func Plotting function to use
#' @param store_scaling Whether to store scaling information for later use
sensitivity_analysis_framework <- function(dat, mods_formula, random_structure,
                                           filters, grouping_var = NULL,
                                           plot = TRUE, plot_func = plot_extreme_filtering,
                                           store_scaling = FALSE) {
    results_list <- list()
    model_info_list <- list()

    # Determine if we're doing grouped or ungrouped analysis
    if (is.null(grouping_var)) {
        groups <- list("All" = dat)
    } else {
        groups <- split(dat, dat[[grouping_var]])
        # names(groups) <- unique(dat[[grouping_var]])
    }
    # order groups by size (largest first)
    groups <- groups[order(sapply(groups, nrow), decreasing = TRUE)]

    # Process each group
    for (group_name in names(groups)) {
        group_data <- groups[[group_name]]

        # Skip groups that are too small
        if (nrow(group_data) < 5) {
            message(sprintf("Skipping group %s: too few observations (%d)", group_name, nrow(group_data)))
            next
        }

        message(sprintf("Processing group: %s", group_name))

        # Fit baseline model
        baseline_model <- safe_fit_model(
            group_data, mods_formula, random_structure,
            sprintf("%s baseline", group_name)
        )
        if (!is.null(baseline_model)) {
            group_label <- if (is.null(grouping_var)) NA else group_name
            results_list[[length(results_list) + 1]] <- extract_results(
                baseline_model, "All data", "None (all data)", NA, group_label, group_data, store_scaling
            )
            model_info_list[[paste0(group_name, "_baseline")]] <- list(
                model = baseline_model, data = group_data
            )
        }

        # Apply each filter
        for (filter_name in names(filters)) {
            filter_func <- filters[[filter_name]]
            filter_results <- filter_func(
                group_data, group_name, baseline_model,
                mods_formula, random_structure
            )

            # Add results from this filter
            if (!is.null(filter_results$results)) {
                results_list <- c(results_list, filter_results$results)
            }
            if (!is.null(filter_results$models)) {
                model_info_list <- c(model_info_list, filter_results$models)
            }
        }
    }

    # Combine and process results
    if (length(results_list) > 0) {
        # Ensure ExclusionValue is character for consistency
        results_list <- lapply(results_list, function(df) {
            if ("ExclusionValue" %in% names(df)) {
                df$ExclusionValue <- as.character(df$ExclusionValue)
            }
            df
        })

        sens_results <- bind_rows(results_list)

        if (plot && nrow(sens_results) > 0) {
            plot_func(sens_results)
        }

        return(list(
            results = sens_results,
            models = model_info_list
        ))
    } else {
        warning("No successful model fits")
        return(list(results = data.frame(), models = list()))
    }
}

# SPECIFIC FILTER FUNCTIONS
# ========================

#' Extreme value percentile filter
extreme_percentile_filter <- function(group_data, group_name, baseline_model, mods_formula, random_structure, percentiles = 1:10, store_scaling = FALSE) {
    results_list <- list()
    model_info_list <- list()

    for (p in percentiles) {
        q_low <- quantile(group_data$yi, p / 100, na.rm = TRUE)
        q_high <- quantile(group_data$yi, 1 - p / 100, na.rm = TRUE)
        dat_trim <- subset(group_data, yi >= q_low & yi <= q_high)
        scenario <- sprintf("Trim extremes (%.0f%%)", p)

        print(sprintf("Fitting model with extremes trimmed at %.0f%% (%.2f, %.2f)...", p, q_low, q_high))

        res_trim <- safe_fit_model(dat_trim, mods_formula, random_structure, scenario)
        if (!is.null(res_trim)) {
            results_list[[length(results_list) + 1]] <- extract_results(
                res_trim, scenario, "Effect size percentile", p, group_name, group_data, store_scaling
            )
            model_info_list[[paste0(group_name, "_trim_", p)]] <- list(
                model = res_trim, data = dat_trim, q_low = q_low, q_high = q_high
            )
        }
    }

    return(list(results = results_list, models = model_info_list))
}

#' Variance percentile filter
variance_percentile_filter <- function(group_data, group_name, baseline_model, mods_formula, random_structure, percentiles = 1:10, store_scaling = FALSE) {
    results_list <- list()
    model_info_list <- list()

    for (p in percentiles) {
        v_thresh <- quantile(group_data$vi, 1 - p / 100, na.rm = TRUE)
        dat_var <- subset(group_data, vi <= v_thresh)
        scenario <- sprintf("High variance (top %.0f%%) excluded", p)

        print(sprintf("Fitting model with high-variance points excluded at %.0f%% (v_thresh=%.2f)...", p, v_thresh))

        res_var <- safe_fit_model(dat_var, mods_formula, random_structure, scenario)
        if (!is.null(res_var)) {
            results_list[[length(results_list) + 1]] <- extract_results(
                res_var, scenario, "Variance percentile", p, group_name, group_data, store_scaling
            )
            model_info_list[[paste0(group_name, "_var_", p)]] <- list(
                model = res_var, data = dat_var, v_thresh = v_thresh
            )
        }
    }

    return(list(results = results_list, models = model_info_list))
}

#' User-defined extreme limits filter
user_limits_filter <- function(group_data, group_name, baseline_model, mods_formula, random_structure, extreme_limits = NULL, store_scaling = FALSE) {
    results_list <- list()
    model_info_list <- list()

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
            dat_lim <- subset(group_data, yi >= lim_low & yi <= lim_high)
            scenario <- sprintf("User limits (%.2f, %.2f)", lim_low, lim_high)

            print(sprintf("Fitting model with user-specified limits: yi in [%.2f, %.2f]...", lim_low, lim_high))

            res_lim <- safe_fit_model(dat_lim, mods_formula, random_structure, scenario)
            if (!is.null(res_lim)) {
                exclusion_value_str <- sprintf("%.2f:%.2f", lim_low, lim_high)
                userlimits_str <- sprintf("User limits (%.1f, %.1f)", lim_low, lim_high)

                results_list[[length(results_list) + 1]] <- extract_results(
                    res_lim, scenario, userlimits_str, exclusion_value_str, group_name, group_data, store_scaling
                )
                model_info_list[[paste0(group_name, "_limits_", length(results_list))]] <- list(
                    model = res_lim, data = dat_lim, lim_low = lim_low, lim_high = lim_high
                )
            }
        }
    }

    return(list(results = results_list, models = model_info_list))
}

#' Cook's distance influence filter
cooks_distance_filter <- function(group_data, group_name, baseline_model, mods_formula, random_structure, store_scaling = FALSE) {
    results_list <- list()
    model_info_list <- list()

    if (is.null(baseline_model)) {
        return(list(results = list(), models = list()))
    }

    n_obs <- nrow(group_data)
    n_coef <- length(coef(baseline_model))

    # Calculate thresholds
    thresholds <- list(
        "conservative" = 4 / (n_obs - n_coef),
        "liberal" = 2 * sqrt(n_coef / (n_obs - n_coef - 1))
    )

    cooks_scenario_labels <- c(
        "conservative" = "Conservative threshold",
        "liberal" = "Liberal threshold"
    )

    cooks <- compute_cooks_distance(baseline_model)

    for (thr_name in names(thresholds)) {
        thr <- thresholds[[thr_name]]
        print(paste("Applying", thr_name, "threshold:", round(thr, 4)))

        to_exclude <- which(cooks > thr)
        scenario_label <- cooks_scenario_labels[[thr_name]]

        if (length(to_exclude) > 0 && length(to_exclude) < nrow(group_data)) {
            dat_cook <- group_data[-to_exclude, ]
            res_cook <- safe_fit_model(dat_cook, mods_formula, random_structure, scenario_label)

            if (!is.null(res_cook)) {
                results_list[[length(results_list) + 1]] <- extract_results(
                    res_cook, scenario_label, "Cook's distance", thr_name, group_name, group_data, store_scaling
                )
                model_info_list[[paste0(group_name, "_", scenario_label)]] <- list(
                    model = res_cook, data = dat_cook, excluded = to_exclude, cooks_threshold = thr
                )
            }
        } else if (length(to_exclude) == 0) {
            message(sprintf("No influential points found for %s threshold in group %s.", thr_name, group_name))
            results_list[[length(results_list) + 1]] <- extract_results(
                baseline_model, scenario_label, "Cook's distance", thr_name, group_name, group_data, store_scaling
            )
        } else {
            message(sprintf("All points would be excluded for %s threshold in group %s; skipping.", thr_name, group_name))
        }
    }

    return(list(results = results_list, models = model_info_list))
}

#' Climate bounds filter
climate_filter <- function(group_data, group_name, baseline_model, mods_formula, random_structure,
                           dt_bounds = c(0, 4), dph_bounds = c(-0.4, 0), store_scaling = FALSE) {
    results_list <- list()
    model_info_list <- list()

    print("Fitting climate filtered model...")
    print(dt_bounds[1])
    dat_clim <- subset(
        group_data,
        dt >= dt_bounds[1] & dt <= dt_bounds[2] &
            dph >= dph_bounds[1] & dph <= dph_bounds[2]
    )

    res_clim <- safe_fit_model(dat_clim, mods_formula, random_structure, "Climate filtered")
    if (!is.null(res_clim)) {
        results_list[[length(results_list) + 1]] <- extract_results(
            res_clim, "Climate filtered", "Climate bounds",
            sprintf("dt[%.1f,%.1f],dph[%.1f,%.1f]", dt_bounds[1], dt_bounds[2], dph_bounds[1], dph_bounds[2]),
            group_name, group_data, store_scaling
        )
        model_info_list[[paste0(group_name, "_clim_filtered")]] <- list(
            model = res_clim, data = dat_clim
        )
    }

    return(list(results = results_list, models = model_info_list))
}

# REFACTORED MAIN FUNCTIONS
# =========================

#' Extreme value filtering sensitivity analysis
#' @param dat Data frame containing meta-analysis data
#' @param mods_formula Model formula for fixed effects
#' @param random_structure Random effects structure
#' @param plot Whether to plot results
#' @param percentiles Vector of percentiles for extreme value trimming
#' @param extreme_limits Optional user-defined extreme limits
#' @param store_scaling Logical. If TRUE, stores scaling information (standard deviations)
#'        for later coefficient rescaling. Set this to TRUE if you plan to use
#'        rescale_coefficients = TRUE in plot_coefficient_comparison().
extreme_filtering <- function(
    dat,
    mods_formula = NULL,
    random_structure,
    plot = TRUE,
    percentiles = 1:10,
    extreme_limits = NULL,
    store_scaling = FALSE,
    rescale_coefficients = FALSE) {
    # Define filters to apply
    filters <- list()

    # Add percentile filters with closures to capture parameters
    filters[["extreme_percentile"]] <- function(group_data, group_name, baseline_model, mods_formula, random_structure) {
        extreme_percentile_filter(group_data, group_name, baseline_model, mods_formula, random_structure, percentiles, store_scaling)
    }

    filters[["variance_percentile"]] <- function(group_data, group_name, baseline_model, mods_formula, random_structure) {
        variance_percentile_filter(group_data, group_name, baseline_model, mods_formula, random_structure, percentiles, store_scaling)
    }

    # Add user limits filter if provided
    if (!is.null(extreme_limits)) {
        filters[["user_limits"]] <- function(group_data, group_name, baseline_model, mods_formula, random_structure) {
            user_limits_filter(group_data, group_name, baseline_model, mods_formula, random_structure, extreme_limits, store_scaling)
        }
    }

    # Create plotting function with rescale_coefficients parameter
    plot_func <- if (plot) {
        function(sens_results) {
            plot_extreme_filtering(sens_results, rescale_coefficients = rescale_coefficients)
        }
    } else {
        NULL
    }

    # Use the framework
    return(sensitivity_analysis_framework(
        dat = dat,
        mods_formula = mods_formula,
        random_structure = random_structure,
        filters = filters,
        grouping_var = NULL,
        plot = plot,
        plot_func = plot_func,
        store_scaling = store_scaling
    ))
}


loo_analysis <- function(dat, mods_formula = NULL, random_structure,
                         study_var = "doi", level = c("sample", "study")) {
    level <- match.arg(level, choices = c("sample", "study"))
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
                           rescale_coefficients = FALSE,
                           plot = TRUE) {
    # Investigate the effect of different st_calcification_unit values by fitting models separately for each unit type
    filters <- list()

    # For each unique st_calcification_unit, define a filter that keeps only that unit
    unit_types <- unique(dat$st_calcification_unit)
    for (unit in unit_types) {
        filters[[as.character(unit)]] <- function(group_data, group_name, baseline_model, mods_formula, random_structure) {
            # Filter group_data to only this unit
            group_data_unit <- subset(group_data, st_calcification_unit == unit)
            if (nrow(group_data_unit) < 2) {
                return(NULL)
            } # skip if too few
            # Fit model
            fit <- tryCatch(
                rma.mv(
                    yi, vi,
                    mods = mods_formula,
                    random = random_structure,
                    data = group_data_unit,
                    method = "REML"
                ),
                error = function(e) NULL
            )
            return(fit)
        }
    }

    # Create plotting function with rescale_coefficients parameter
    plot_func <- if (plot) {
        function(sens_results) {
            plot_coefficient_comparison(sens_results, rescale_coefficients = rescale_coefficients)
        }
    } else {
        NULL
    }
    if (rescale_coefficients) {
        store_scaling <- TRUE
    } else {
        store_scaling <- FALSE
    }

    return(sensitivity_analysis_framework(
        dat = dat,
        mods_formula = mods_formula,
        random_structure = random_structure,
        filters = filters,
        store_scaling = store_scaling,
        grouping_var = grouping_var,
        plot = plot,
        plot_func = plot_func
    ))
}


influence_filtering <- function(dat,
                                mods_formula = NULL,
                                random_structure,
                                grouping_var = "core_grouping",
                                rescale_coefficients = FALSE,
                                plot = TRUE) {
    filters <- list()
    filters[["cooks_distance"]] <- function(group_data, group_name, baseline_model, mods_formula, random_structure) {
        cooks_distance_filter(group_data, group_name, baseline_model, mods_formula, random_structure)
    }

    # Create plotting function with rescale_coefficients parameter
    plot_func <- if (plot) {
        function(sens_results) {
            plot_coefficient_comparison(sens_results, rescale_coefficients = rescale_coefficients)
        }
    } else {
        NULL
    }

    return(sensitivity_analysis_framework(
        dat = dat,
        mods_formula = mods_formula,
        random_structure = random_structure,
        filters = filters,
        grouping_var = grouping_var,
        plot = plot,
        plot_func = plot_func
    ))
}

clim_filtering <- function(dat,
                           mods_formula = NULL,
                           random_structure,
                           grouping_var = "core_grouping",
                           dt_bounds = c(-2, 4),
                           dph_bounds = c(-0.4, 0),
                           rescale_coefficients = FALSE,
                           plot = TRUE) {
    filters <- list()
    filters[["climate"]] <- function(group_data, group_name, baseline_model, mods_formula, random_structure) {
        climate_filter(group_data, group_name, baseline_model, mods_formula, random_structure, dt_bounds, dph_bounds)
    }

    # Create plotting function with rescale_coefficients parameter
    plot_func <- if (plot) {
        function(sens_results) {
            plot_coefficient_comparison(sens_results, rescale_coefficients = rescale_coefficients)
        }
    } else {
        NULL
    }

    if (rescale_coefficients) {
        store_scaling <- TRUE
    } else {
        store_scaling <- FALSE
    }

    return(sensitivity_analysis_framework(
        dat = dat,
        mods_formula = mods_formula,
        random_structure = random_structure,
        filters = filters,
        store_scaling = store_scaling,
        grouping_var = grouping_var,
        plot = plot,
        plot_func = plot_func
    ))
}

#' Test scaling effects on model coefficients
#' @param dat Data frame containing meta-analysis data
#' @param mods_formula Model formula for fixed effects (should use raw variable names)
#' @param random_structure Random effects structure
#' @param grouping_var Optional grouping variable
#' @param plot Whether to plot results
#' @param title Plot title
#' @return List containing results and models
scaling_effects_test <- function(dat,
                                 mods_formula = NULL,
                                 random_structure,
                                 grouping_var = NULL,
                                 plot = TRUE,
                                 title = "Scaling Effects Comparison") {
    # Ensure we have scaled variables available
    cat("Debug: Starting scaling_effects_test\n")
    if (!"dt_s" %in% colnames(dat) || !"dph_s" %in% colnames(dat)) {
        # Create scaled variables if they don't exist
        if ("dt" %in% colnames(dat)) {
            dat$dt_s <- dat$dt / sd(dat$dt, na.rm = TRUE)
        } else if ("delta_t" %in% colnames(dat)) {
            dat$dt <- dat$delta_t
            dat$dt_s <- dat$dt / sd(dat$dt, na.rm = TRUE)
        }

        if ("dph" %in% colnames(dat)) {
            dat$dph_s <- dat$dph / sd(dat$dph, na.rm = TRUE)
        } else if ("delta_ph" %in% colnames(dat)) {
            dat$dph <- dat$delta_ph
            dat$dph_s <- dat$dph / sd(dat$dph, na.rm = TRUE)
        }
    }
    cat("Debug: Created scaled variables\n")
    # Store scaling factors for back-transformation
    sd_dt <- sd(dat$dt, na.rm = TRUE)
    sd_dph <- sd(dat$dph, na.rm = TRUE)

    # Convert formula to use scaled variables
    formula_text <- deparse(mods_formula)
    formula_text <- gsub("\\bdt\\b", "dt_s", formula_text)
    formula_text <- gsub("\\bdph\\b", "dph_s", formula_text)
    scaled_formula <- as.formula(formula_text)

    cat("Original formula:", deparse(mods_formula), "\n")
    cat("Scaled formula:", deparse(scaled_formula), "\n")
    results_list <- list()
    model_info_list <- list()

    cat("Debug: Converted formula to scaled version\n")
    # Determine if we're doing grouped or ungrouped analysis
    if (is.null(grouping_var)) {
        groups <- list("All" = dat)
    } else {
        groups <- split(dat, dat[[grouping_var]])
    }

    # Process each group
    for (group_name in names(groups)) {
        group_data <- groups[[group_name]]

        # Skip groups that are too small
        if (nrow(group_data) < 5) {
            message(sprintf("Skipping group %s: too few observations (%d)", group_name, nrow(group_data)))
            next
        }

        message(sprintf("Processing group: %s", group_name))
        # Fit model with raw variables
        raw_model <- safe_fit_model(
            group_data, mods_formula, random_structure,
            sprintf("%s raw model", group_name)
        )

        # Fit model with scaled variables
        scaled_model <- safe_fit_model(
            group_data, scaled_formula, random_structure,
            sprintf("%s scaled model", group_name)
        )
        cat("Debug: Fitted models for group:", group_name, "\n")
        if (!is.null(raw_model)) {
            group_label <- if (is.null(grouping_var)) "All" else group_name
            results_list[[length(results_list) + 1]] <- extract_results(
                raw_model, "Raw model coefficients", "Scaling comparison", "raw", group_label, group_data, TRUE
            )
            model_info_list[[paste0(group_name, "_raw")]] <- list(
                model = raw_model, data = group_data
            )
        }
        cat("Debug: Processing raw model results\n")
        if (!is.null(scaled_model)) {
            group_label <- if (is.null(grouping_var)) "All" else group_name
            # Extract results and apply back-transformation
            scaled_results <- extract_results(
                scaled_model, "Scaled model coefficients", "Scaling comparison", "scaled", group_label, group_data, TRUE
            )
            cat("Debug: Processing scaled model results\n")
            # Back-transform scaled coefficients to raw units
            for (i in seq_len(nrow(scaled_results))) {
                term <- scaled_results$Term[i]
                scale_factor <- 1 # Default: no scaling

                # Determine scaling factor based on term name
                if (grepl("dt_s", term)) {
                    scale_factor <- sd_dt
                } else if (grepl("dph_s", term)) {
                    scale_factor <- sd_dph
                } else if (grepl("I\\(dt_s\\^2\\)", term)) {
                    scale_factor <- sd_dt^2
                } else if (grepl("I\\(dph_s\\^2\\)", term)) {
                    scale_factor <- sd_dph^2
                } else if (grepl("dt_s:dph_s", term)) {
                    scale_factor <- sd_dt * sd_dph
                } else if (grepl("I\\(dt_s\\^2\\):I\\(dph_s\\^2\\)", term)) {
                    scale_factor <- sd_dt^2 * sd_dph^2
                }

                # Apply back-transformation
                if (scale_factor != 1) {
                    scaled_results$est[i] <- scaled_results$est[i] / scale_factor
                    scaled_results$ci.lb[i] <- scaled_results$ci.lb[i] / scale_factor
                    scaled_results$ci.ub[i] <- scaled_results$ci.ub[i] / scale_factor

                    # Update coefficient name to remove scaling indicators
                    scaled_results$Term[i] <- unscale_coefficient_names(term)
                }
            }

            results_list[[length(results_list) + 1]] <- scaled_results
            model_info_list[[paste0(group_name, "_scaled")]] <- list(
                model = scaled_model, data = group_data
            )
        }
    }
    cat("Debug: Length of results_list:", length(results_list), "\n")
    # Combine results
    if (length(results_list) > 0) {
        sens_results <- bind_rows(results_list)

        # if (plot && nrow(sens_results) > 0) {
        #     plot_coefficient_comparison(sens_results, title = title)
        # }

        return(list(
            results = sens_results,
            models = model_info_list,
            scaling_factors = list(sd_dt = sd_dt, sd_dph = sd_dph)
        ))
    } else {
        warning("No successful model fits")
        return(list(results = data.frame(), models = list(), scaling_factors = list()))
    }
}

#' Plot scaling comparison results with difference calculations
#' @param scaling_results Results from scaling_effects_test function
#' @param title Plot title
#' @param show_differences Whether to print coefficient differences
#' @return ggplot object
plot_scaling_comparison <- function(scaling_results, title = "Scaling Effects Comparison", show_differences = TRUE, width = 10, height = 6, return_plot = FALSE) {
    # if (is.null(scaling_results$results) || nrow(scaling_results$results) == 0) {
    #     stop("No results to plot")
    # }

    sens_results <- scaling_results$results

    # Calculate differences between raw and scaled models for each coefficient
    if (show_differences) {
        # Get unique terms and groups
        unique_terms <- unique(sens_results$Term)
        unique_groups <- unique(sens_results$Group)

        cat("\nCoefficient Differences (Raw - Scaled):\n")
        cat("=======================================\n")

        for (group in unique_groups) {
            if (!is.na(group)) cat(sprintf("\nGroup: %s\n", group))

            for (term in unique_terms) {
                raw_coef <- sens_results[sens_results$Term == term &
                    sens_results$Scenario == "Raw model coefficients" &
                    sens_results$Group == group, "est"]
                scaled_coef <- sens_results[sens_results$Term == term &
                    sens_results$Scenario == "Scaled model coefficients" &
                    sens_results$Group == group, "est"]

                if (length(raw_coef) > 0 && length(scaled_coef) > 0) {
                    diff <- raw_coef - scaled_coef
                    percent_diff <- abs(diff / raw_coef) * 100

                    cat(sprintf(
                        "  %s: %.6f (%.2f%% difference)\n",
                        term, diff, percent_diff
                    ))
                }
            }
        }
        cat("\n")
    }

    # Create the plot using the standard plotting function
    if (return_plot) {
        return(plot_coefficient_comparison(sens_results, title = title, width = width, height = height, return_plot = TRUE))
    } else {
        plot_coefficient_comparison(sens_results, title = title, width = width, height = height, return_plot = FALSE)
    }
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
        return(round(vifs, 0))
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
    # create sequence of dt and dph between their minimum and maximum values in the data
    dt_seq <- seq(min(dat[[varnames$dt]], na.rm = TRUE), max(dat[[varnames$dt]], na.rm = TRUE), length = n_points)
    dph_seq <- seq(min(dat[[varnames$dph]], na.rm = TRUE), max(dat[[varnames$dph]], na.rm = TRUE), length = n_points)
    grid <- expand.grid(dt = dt_seq, dph = dph_seq)

    mean_dt <- mean(dat[[varnames$dt]], na.rm = TRUE)
    mean_dph <- mean(dat[[varnames$dph]], na.rm = TRUE)
    sd_dt <- sd(dat[[varnames$dt]], na.rm = TRUE)
    sd_dph <- sd(dat[[varnames$dph]], na.rm = TRUE)

    # Add all possible transformations to handle different model types
    # Centered versions (mean-subtracted)
    grid$dt_c <- grid$dt - mean_dt
    grid$dph_c <- grid$dph - mean_dph

    # Scaled-only versions (divided by SD, not centered)
    grid$dt_s <- grid$dt / sd_dt
    grid$dph_s <- grid$dph / sd_dph

    # Standardized versions (centered and scaled)
    grid$dt_z <- (grid$dt - mean_dt) / sd_dt
    grid$dph_z <- (grid$dph - mean_dph) / sd_dph

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
                            squadnames = list(dt2 = "I(dt_s^2)", dph2 = "I(dph_s^2)"),
                            zquadnames = list(dt2 = "I(dt_z^2)", dph2 = "I(dph_z^2)"),
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
    Xpred <- model.matrix(mods_formula, data = grid)
    # Predicted values
    pred <- as.vector(Xpred %*% b_unit)
    pred_se <- sqrt(diag(Xpred %*% V_unit %*% t(Xpred)))
    pred_lo <- pred - 1.96 * pred_se
    pred_hi <- pred + 1.96 * pred_se

    grid$pred <- pred
    grid$pred_se <- pred_se
    grid$pred_lo <- pred_lo
    grid$pred_hi <- pred_hi

    if (plot) {
        plot_surface_contour(grid, pred, title = title, fig_width = fig_width, fig_height = fig_height, plot_anomalies = plot_anomalies, fill_limits = fill_limits)
    }
    invisible(grid)
    return(list(grid = grid, pred = pred))
}


# PLOTTING

plot_loo <- function(loo_df, full_model = NULL, show = TRUE) {
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

    if (show) {
        print(p_combined)
    }
    return(p_combined)
}


#' Plot coefficient comparison across sensitivity analysis scenarios
#' @param sens_results Data frame containing sensitivity analysis results
#' @param title Plot title
#' @param width Plot width
#' @param height Plot height
#' @param rescale_coefficients Logical. If TRUE and scaling information is available,
#'        converts scaled coefficients back to original units by dividing by appropriate
#'        standard deviations. Requires that sens_results was generated with store_scaling = TRUE.
plot_coefficient_comparison <- function(sens_results, title = "", width = 8, height = 6, rescale_coefficients = FALSE, return_plot = FALSE) {
    # Apply coefficient rescaling if requested
    if (rescale_coefficients) {
        # Check if scaling information is available
        has_scaling_info <- "scaling_available" %in% colnames(sens_results) &&
            any(sens_results$scaling_available == TRUE, na.rm = TRUE)

        # If no scaling info stored, try to auto-detect scaled terms
        if (!has_scaling_info) {
            scaled_terms <- detect_scaled_terms(sens_results$Term)
            if (any(scaled_terms)) {
                warning("Scaled terms detected but no scaling information available. Cannot rescale coefficients.")
                warning("Use store_scaling = TRUE when generating sens_results to enable rescaling.")
            }
        } else {
            scaling_rows <- which(sens_results$scaling_available == TRUE &
                !is.na(sens_results$sd_dt) & !is.na(sens_results$sd_dph))

            if (length(scaling_rows) > 0) {
                message("Rescaling coefficients to original units...")

                for (i in scaling_rows) {
                    term <- sens_results$Term[i]
                    sd_dt <- sens_results$sd_dt[i]
                    sd_dph <- sens_results$sd_dph[i]

                    # Determine scaling factor based on term name
                    scale_factor <- 1 # Default: no scaling

                    # Linear terms
                    if (grepl("dt_z|dt_s", term)) {
                        scale_factor <- sd_dt
                    } else if (grepl("dph_z|dph_s", term)) {
                        scale_factor <- sd_dph
                    }

                    # Quadratic terms
                    else if (grepl("I\\(dt_z\\^2\\)|I\\(dt_s\\^2\\)", term)) {
                        scale_factor <- sd_dt^2
                    } else if (grepl("I\\(dph_z\\^2\\)|I\\(dph_s\\^2\\)", term)) {
                        scale_factor <- sd_dph^2
                    }

                    # Interaction terms
                    else if (grepl("dt_z:dph_z|dt_s:dph_s", term)) {
                        scale_factor <- sd_dt * sd_dph
                    } else if (grepl("I\\(dt_z\\^2\\):I\\(dph_z\\^2\\)|I\\(dt_s\\^2\\):I\\(dph_s\\^2\\)", term)) {
                        scale_factor <- sd_dt^2 * sd_dph^2
                    }

                    # Apply scaling if a scaling factor was determined
                    if (scale_factor != 1) {
                        sens_results$est[i] <- sens_results$est[i] / scale_factor
                        sens_results$ci.lb[i] <- sens_results$ci.lb[i] / scale_factor
                        sens_results$ci.ub[i] <- sens_results$ci.ub[i] / scale_factor

                        # Update coefficient name to remove scaling indicators
                        original_term <- term
                        new_term <- unscale_coefficient_names(term)
                        sens_results$Term[i] <- new_term

                        message(sprintf("Rescaled %s by factor %.3f and renamed to %s", original_term, scale_factor, new_term))
                    }
                }
            } else {
                warning("Rescaling requested but no valid scaling information available in sens_results")
            }
        }
    }

    # Reduce vertical spacing between points/bars
    scenario_spacing <- 0.15

    # Convert Term to numeric for positioning with tighter spacing
    unique_terms <- unique(sens_results$Term)
    n_terms <- length(unique_terms)

    # Use compressed spacing instead of integer spacing
    term_spacing <- 0.8 # Reduce from default 1.0 spacing
    term_positions <- seq(1, by = term_spacing, length.out = n_terms)
    names(term_positions) <- unique_terms

    sens_results$Term_num <- term_positions[sens_results$Term]

    # Calculate vertical offsets for each coefficient term within each group
    sens_results$y_offset <- sens_results$Term_num

    for (group in unique(sens_results$Group)) {
        group_data <- sens_results[sens_results$Group == group, ]

        for (term in unique(group_data$Term)) {
            term_data <- group_data[group_data$Term == term, ]
            term_num <- unique(term_data$Term_num)

            # Get unique scenarios for this term within this group
            scenarios <- unique(term_data$Scenario)
            n_scenarios <- length(scenarios)

            if (n_scenarios > 1) {
                # Center the scenarios around the base term position
                scenario_positions <- seq(-scenario_spacing * (n_scenarios - 1) / 2,
                    scenario_spacing * (n_scenarios - 1) / 2,
                    length.out = n_scenarios
                )
                names(scenario_positions) <- scenarios

                # Apply offsets
                for (scenario in scenarios) {
                    row_idx <- which(sens_results$Group == group &
                        sens_results$Term == term &
                        sens_results$Scenario == scenario)

                    sens_results$y_offset[row_idx] <- term_num + scenario_positions[scenario]
                }
            } else {
                # Single scenario, just use term position
                row_idx <- which(sens_results$Group == group & sens_results$Term == term)
                sens_results$y_offset[row_idx] <- term_num
            }
        }
    }

    # Determine filtering type and create appropriate color scheme and legend
    exclusion_types <- unique(sens_results$ExclusionType)
    scenarios <- unique(sens_results$Scenario)

    # Create color palette and legend title based on filtering type
    if ("Cook's distance" %in% exclusion_types) {
        # Cook's distance filtering
        color_values <- c(
            "Conservative threshold" = "#E31A1C", # red
            "Liberal threshold" = "#1F78B4", # blue
            "All data" = "#2CA02C" # green
        )
        legend_title <- "Cook's Distance\nThreshold"
        sens_results$Group_label <- NA_character_
        for (grp in unique(sens_results$Group)) {
            n_full <- unique(sens_results[sens_results$Group == grp & sens_results$Scenario == "All data", ]$k)
            n_liberal <- unique(sens_results[sens_results$Group == grp & sens_results$Scenario == "Liberal threshold", ]$k)
            n_conservative <- unique(sens_results[sens_results$Group == grp & sens_results$Scenario == "Conservative threshold", ]$k)
            label <- sprintf(
                "%s (Total n = %.0f | Liberal filtered n = %.0f (%.1f%%) | Conservative filtered n = %.0f (%.1f%%))",
                grp,
                n_full,
                n_liberal,
                100 * (n_full - n_liberal) / n_full,
                n_conservative,
                100 * (n_full - n_conservative) / n_full
            )
            idx <- which(sens_results$Group == grp)
            sens_results$Group_label[idx] <- label
        }
    } else if (any(c("Effect size percentile", "Variance percentile") %in% exclusion_types)) {
        n_scenarios <- length(scenarios)
        color_values <- scales::viridis_d(n_scenarios, option = "plasma")
        names(color_values) <- scenarios
        legend_title <- "Filtering\nLevel"
    } else if ("Climate bounds" %in% exclusion_types) {
        color_values <- c(
            "Climate filtered" = "#FF7F00", # orange
            "All data" = "#2CA02C" # green
        )
        legend_title <- "Climate\nBounds"
        sens_results$Group_label <- NA
        for (grp in unique(sens_results$Group)) {
            n_full <- unique(sens_results[sens_results$Group == grp & sens_results$Scenario == "All data", ]$k)
            n_clim_filtered <- unique(sens_results[sens_results$Group == grp & sens_results$Scenario == "Climate filtered", ]$k)
            idx <- which(sens_results$Group == grp)
            sens_results$Group_label[idx] <- sprintf(
                "%s (Total n = %.0f | Post climate filtering n = %.0f | Removed %.1f%%)",
                grp,
                n_full,
                n_clim_filtered,
                100 * (n_full - n_clim_filtered) / n_full
            )
        }
    } else if ("Scaling comparison" %in% exclusion_types) {
        color_values <- c(
            "Raw model coefficients" = "#1F78B4", # blue
            "Scaled model coefficients" = "#E31A1C" # red
        )
        legend_title <- "Model\nType"
        sens_results$Group_label <- NA_character_
        for (grp in unique(sens_results$Group)) {
            n_obs <- unique(sens_results[sens_results$Group == grp, ]$k)[1]
            idx <- which(sens_results$Group == grp)
            sens_results$Group_label[idx] <- sprintf(
                "%s (n = %.0f)",
                grp,
                n_obs
            )
        }
    } else {
        n_scenarios <- length(scenarios)
        color_values <- scales::brewer_pal(type = "qual", palette = "Set1")(n_scenarios)
        names(color_values) <- scenarios
        legend_title <- "Analysis\nType"
        for (grp in unique(sens_results$Group)) {
            n_full <- unique(sens_results[sens_results$Group == grp & sens_results$Scenario == "All data", ]$k)
            idx <- which(sens_results$Group == grp)
            sens_results$Group_label[idx] <- sprintf(
                "%s (Total n = %.0f)",
                grp,
                n_full
            )
        }
    }

    # Ensure all scenarios in data have colors
    missing_scenarios <- setdiff(scenarios, names(color_values))
    if (length(missing_scenarios) > 0) {
        additional_colors <- rainbow(length(missing_scenarios))
        names(additional_colors) <- missing_scenarios
        color_values <- c(color_values, additional_colors)
    }

    # Ensure Group_label is a factor with levels in the order of appearance in the data
    sens_results$Group_label <- factor(sens_results$Group_label, levels = unique(sens_results$Group_label))

    p <- ggplot(sens_results, aes(x = est, y = y_offset)) +
        geom_point(aes(color = Scenario), size = 2) +
        geom_errorbar(
            aes(xmin = ci.lb, xmax = ci.ub, color = Scenario),
            width = 0.02
        ) +
        geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5) +
        facet_wrap(~Group_label, ncol = 1, scales = "free_y") +
        scale_y_continuous(
            breaks = unique(sens_results$Term_num),
            labels = unique(sens_results$Term),
            name = "Model Coefficient",
            limits = c(min(sens_results$y_offset) - 0.3, max(sens_results$y_offset) + 0.3),
            expand = expansion(mult = 0.02)
        ) +
        scale_color_manual(
            values = color_values,
            name = legend_title
        ) +
        labs(
            x = "Coefficient Estimate (±95% CI)",
            title = title
        ) +
        theme_minimal(base_size = 12) +
        theme(
            panel.grid.minor.y = element_blank(),
            strip.background = element_rect(fill = "grey90", colour = NA),
            strip.text = element_text(face = "bold"),
            legend.position = "bottom"
        )

    if (return_plot) {
        return(p)
    } else {
        print(p)
    }
}

plot_cooks_distance <- function(influence_df, model, fig_height = 8, fig_width = 12, main = NULL, file = NULL, subplot = FALSE) {
    # If subplot mode, skip device management
    if (!subplot) {
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
            legend.position = "bottom",
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
plot_extreme_filtering <- function(sens_results, title = "", rescale_coefficients = FALSE) {
    # Apply coefficient rescaling if requested (same logic as plot_coefficient_comparison)
    if (rescale_coefficients) {
        # Check if scaling information is available
        has_scaling_info <- "scaling_available" %in% colnames(sens_results) &&
            any(sens_results$scaling_available == TRUE, na.rm = TRUE)

        # If no scaling info stored, try to auto-detect scaled terms
        if (!has_scaling_info) {
            scaled_terms <- detect_scaled_terms(sens_results$Term)
            if (any(scaled_terms)) {
                warning("Scaled terms detected but no scaling information available. Cannot rescale coefficients.")
                warning("Use store_scaling = TRUE when generating sens_results to enable rescaling.")
            }
        } else {
            scaling_rows <- which(sens_results$scaling_available == TRUE &
                !is.na(sens_results$sd_dt) & !is.na(sens_results$sd_dph))

            if (length(scaling_rows) > 0) {
                message("Rescaling coefficients to original units...")

                for (i in scaling_rows) {
                    term <- sens_results$Term[i]
                    sd_dt <- sens_results$sd_dt[i]
                    sd_dph <- sens_results$sd_dph[i]

                    # Determine scaling factor based on term name
                    scale_factor <- 1 # Default: no scaling

                    # Linear terms
                    if (grepl("dt_z|dt_s", term)) {
                        scale_factor <- sd_dt
                    } else if (grepl("dph_z|dph_s", term)) {
                        scale_factor <- sd_dph
                    }

                    # Quadratic terms
                    else if (grepl("I\\(dt_z\\^2\\)|I\\(dt_s\\^2\\)", term)) {
                        scale_factor <- sd_dt^2
                    } else if (grepl("I\\(dph_z\\^2\\)|I\\(dph_s\\^2\\)", term)) {
                        scale_factor <- sd_dph^2
                    }

                    # Interaction terms
                    else if (grepl("dt_z:dph_z|dt_s:dph_s", term)) {
                        scale_factor <- sd_dt * sd_dph
                    } else if (grepl("I\\(dt_z\\^2\\):I\\(dph_z\\^2\\)|I\\(dt_s\\^2\\):I\\(dph_s\\^2\\)", term)) {
                        scale_factor <- sd_dt^2 * sd_dph^2
                    }

                    # Apply scaling if a scaling factor was determined
                    if (scale_factor != 1) {
                        sens_results$est[i] <- sens_results$est[i] / scale_factor
                        sens_results$ci.lb[i] <- sens_results$ci.lb[i] / scale_factor
                        sens_results$ci.ub[i] <- sens_results$ci.ub[i] / scale_factor

                        # Update coefficient name to remove scaling indicators
                        original_term <- term
                        new_term <- unscale_coefficient_names(term)
                        sens_results$Term[i] <- new_term

                        message(sprintf("Rescaled %s by factor %.3f and renamed to %s", original_term, scale_factor, new_term))
                    }
                }
            } else {
                warning("Rescaling requested but no valid scaling information available in sens_results")
            }
        }
    }

    # Create manual vertical offsets for each ExclusionType and ExclusionValue combination

    type_spacing <- 0.25 # Spacing between different ExclusionTypes
    value_spacing <- 0.02 # Spacing between different ExclusionValues within the same type

    # drop effect size percentile exclusion type
    sens_results <- sens_results[sens_results$ExclusionType != "Effect size percentile", ]

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
            name = "Exclusion Level (top n %)",
            na.value = "grey50"
        ) +
        scale_y_continuous(
            breaks = unique(sens_results$Term_num),
            labels = unique(sens_results$Term),
            name = "Coefficient",
            limits = c(min(sens_results$y_offset) - 0.3, max(sens_results$y_offset) + 0.3),
            expand = expansion(mult = 0.02)
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


plot_cooks_distance_by_group <- function(dat, mods_formula, random_structure, grouping_var = "core_grouping", ncpus = 64, parallel = "multicore", fig_height = 5, fig_width = 10, file_prefix = NULL, combined_plot = TRUE) {
    # Fit model to all data and compute influence measures
    fit_all <- rma.mv(
        yi = yi, V = vi, mods = mods_formula,
        random = random_structure, data = dat, method = "REML"
    )
    inf_measures_all <- compute_influence_measures(fit_all, dat, parallel = parallel, ncpus = ncpus)
    inf_measures_all[[grouping_var]] <- "All"

    # For each group, fit model and compute influence
    group_levels <- unique(tolower(dat[[grouping_var]]))
    influence_by_group <- list()
    fit_by_group <- list()
    for (cg in group_levels) {
        dat_cg <- subset(dat, tolower(dat[[grouping_var]]) == cg)
        if (nrow(dat_cg) < 2) next # skip if not enough data
        message(sprintf("Processing %s: %s rows", cg, nrow(dat_cg)))
        fit_cg <- rma.mv(
            yi = yi, V = vi, mods = mods_formula,
            random = random_structure, data = dat_cg, method = "REML"
        )
        inf_measures_cg <- compute_influence_measures(fit_cg, dat_cg, parallel = parallel, ncpus = ncpus)
        inf_measures_cg[[grouping_var]] <- cg
        influence_by_group[[cg]] <- inf_measures_cg
        fit_by_group[[cg]] <- fit_cg
    }

    if (combined_plot) {
        # Create combined plot with all groups in subplots
        n_plots <- 1 + length(influence_by_group) # "All" + group plots

        # Set up file device if file_prefix is provided
        if (!is.null(file_prefix)) {
            combined_file <- sprintf("%s_combined.png", file_prefix)
            png(filename = combined_file, width = fig_width, height = fig_height * n_plots, units = "in", res = 300)
            on.exit(dev.off(), add = TRUE)
        } else {
            # For interactive mode, create new device with appropriate height
            if (dev.cur() == 1) {
                dev.new(width = fig_width, height = fig_height * n_plots)
                on.exit(dev.off(), add = TRUE)
            }
        }

        # Set up subplot layout (n_plots rows, 1 column)
        old_par <- par(mfrow = c(n_plots, 1), mar = c(4, 4, 3, 2) + 0.1)
        on.exit(par(old_par), add = TRUE)

        # Plot "All data" first
        main_title <- "Cook's distance: All data"
        plot_cooks_distance(inf_measures_all, fit_all, main = main_title, subplot = TRUE)

        # Plot each group
        for (cg in names(influence_by_group)) {
            inf_df <- influence_by_group[[cg]]
            fit_cg <- fit_by_group[[cg]]
            main_title <- sprintf("Cook's distance: %s", cg)
            plot_cooks_distance(inf_df, fit_cg, main = main_title, subplot = TRUE)
        }
    } else {
        # Original behavior: individual plots
        # Plot Cook's distance for all data
        if (is.null(file_prefix)) {
            main_title <- sprintf("Cook's distance: All data")
            plot_cooks_distance(inf_measures_all, fit_all, fig_height = fig_height, fig_width = fig_width, main = main_title)
        } else {
            file <- sprintf("%s_All.png", file_prefix)
            main_title <- sprintf("Cook's distance: All data")
            plot_cooks_distance(inf_measures_all, fit_all, fig_height = fig_height, fig_width = fig_width, main = main_title, file = file)
        }

        # Plot Cook's distance for each group
        for (cg in names(influence_by_group)) {
            inf_df <- influence_by_group[[cg]]
            fit_cg <- fit_by_group[[cg]]
            if (is.null(file_prefix)) {
                main_title <- sprintf("Cook's distance: %s", cg)
                plot_cooks_distance(inf_df, fit_cg, fig_height = fig_height, fig_width = fig_width, main = main_title)
            } else {
                file <- sprintf("%s_%s.png", file_prefix, cg)
                main_title <- sprintf("Cook's distance: %s", cg)
                plot_cooks_distance(inf_df, fit_cg, fig_height = fig_height, fig_width = fig_width, main = main_title, file = file)
            }
        }
    }

    invisible(list(
        influence_by_group = influence_by_group,
        all_influence = inf_measures_all
    ))
}
