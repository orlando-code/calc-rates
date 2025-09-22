# Required package
library(metafor)
library(dplyr)
library(parallel)
library(ggplot2)
library(dplyr)
library(tidyr)
library(httpgd)


source("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/calcification/analysis/rnative/new_helpers.R")

# ----------------------------
# 0. Data preparation
# ----------------------------
### load data
dat <- read.csv("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/data/clean/analysis_ready_data.csv")
# remove rows with NA in yi or vi
dat <- dat[!is.na(dat$st_relative_calcification) & !is.na(dat$st_relative_calcification_var), ]
# remove extreme climatology
dat <- dat[dat$delta_t < 4.7, ]
dat <- dat[dat$delta_ph > -0.445, ]

# rename effect sizes and their variance for convenience
colnames(dat)[which(colnames(dat) == "st_relative_calcification")] <- "yi"
colnames(dat)[which(colnames(dat) == "st_relative_calcification_var")] <- "vi"
# rename key moderators for convenience
dat$dt <- dat$delta_t
dat$dph <- dat$delta_ph

### transform moderators
# centre only
dat$dt_c <- dat$delta_t - mean(dat$delta_t, na.rm = TRUE)
dat$dph_c <- dat$delta_ph - mean(dat$delta_ph, na.rm = TRUE)
# scale only
dat$dt_s <- dat$delta_t / sd(dat$delta_t, na.rm = TRUE)
dat$dph_s <- dat$delta_ph / sd(dat$delta_ph, na.rm = TRUE)
# z-scale (centre and scale)
dat$dt_z <- (dat$delta_t - mean(dat$delta_t, na.rm = TRUE)) / sd(dat$delta_t, na.rm = TRUE)
dat$dph_z <- (dat$delta_ph - mean(dat$delta_ph, na.rm = TRUE)) / sd(dat$delta_ph, na.rm = TRUE)



dat_small <- dat[1:100, ]
# chosen random structure (due to AIC and physical sense)
random_structure <- ~ 1 | doi / ID + 1 | species_types
# mods_formula <- ~ dt + dph + dt:dph
mods_formula <- ~ dt + dph - 1


### formulae bucket
mods_formula <- ~ delta_t + delta_ph + I(delta_t^2) + I(delta_ph^2) + delta_t:delta_ph - 1
cg_mods_formula <- ~ dt_c + dph_c + I(dt_c^2) + I(dph_c^2) + dt_c:dph_c - 1
cg_mods_formula <- ~ dt_z + dph_z + I(dt_z^2) + I(dph_z^2) + dt_z:dph_z + I(dt_z^2):I(dph_z^2) - 1
mods_formula <- ~ dt_z + dph_z + dt_z:dph_z - 1
mods_formula <- ~ dt_z + dph_z + dt_z:dph_z
mods_formula <- ~ dt_z + dph_z - 1


# -------------
# Exploring random structures
# -------------
doi_fit <- rma.mv(
    yi = yi, V = vi, mods = mods_formula,
    random = ~ 1 | doi, data = dat, method = "REML"
)
doi_id_fit <- rma.mv(
    yi = yi, V = vi, mods = mods_formula,
    random = ~ 1 | doi / ID, data = dat, method = "REML"
)
doi_cg_irr_fit <- rma.mv(
    yi = yi, V = vi, mods = mods_formula,
    random = ~ 1 | doi / core_grouping / irr_group, data = dat, method = "REML"
)
summary(doi_fit)
summary(doi_id_fit)
summary(doi_cg_irr_fit)


# fit all with treatment type as a moderator
overall_model_formula <- ~ dt_s + dph_s + dt_s:dph_s - 1
overall_model_formula <- ~ dt + dph + dt:dph - 1
# overall_model_formula <- ~ dt_s + dph_s - 1
overall_model <- rma.mv(
    yi = yi, V = vi, mods = overall_model_formula,
    random = list(~1 | doi / ID), data = dat, method = "REML"
)
summary(overall_model)

library(MuMIn)
library(parallel)
eval(metafor:::.MuMIn)

# Function to run dredge and summarize results for a single core_grouping
run_dredge_for_cg <- function(cg, dat, overall_model_formula) {
    cat("\n--- core_grouping:", cg, "---\n")
    dat_cg <- subset(dat, core_grouping == cg)
    # dat_cg <- dat_cg[1:20, ]
    if (nrow(dat_cg) < 2) {
        cat("Not enough data for this core_grouping.\n")
        return(NULL)
    }
    # Fit the base model
    model_cg <- tryCatch(
        rma.mv(
            yi = yi, V = vi, mods = overall_model_formula,
            random = list(~1 | doi / ID), data = dat_cg, method = "REML"
        ),
        error = function(e) {cat("Model failed:", e$message, "\n"); return(NULL)}
    )
    if (is.null(model_cg)) return(NULL)

    # Convert to MuMIn-compatible object
    class(model_cg) <- c("rma.mv", class(model_cg))
    # Dredge all subsets of moderators
    dd <- tryCatch(
        dredge(model_cg, trace = FALSE, rank = "AICc"),
        error = function(e) {cat("Dredge failed:", e$message, "\n"); return(NULL)}
    )
    if (is.null(dd)) return(NULL)

    # Show top 10 models
    print(dd[1:min(10, nrow(dd)), ])

    # For each moderator, calculate proportion of models where p < 0.05
    # Get moderator names
    mod_names <- names(dd)[grepl("^dt|^dph", names(dd))]
    sig_props <- sapply(mod_names, function(mod) {
        # Find models where moderator is included (not NA)
        included <- !is.na(dd[[mod]])
        if (!any(included)) return(NA)
        # For those, get p-values
        pvals <- dd[included, paste0("p.", mod)]
        # Proportion with p < 0.05
        mean(pvals < 0.05, na.rm = TRUE)
    })
    cat("Proportion of models where each moderator is significant (p < 0.05):\n")
    print(cg)
    print(summary(model_cg))
    print(sig_props)
    invisible(list(dredge = dd, sig_props = sig_props))
}

other_algae_dat <- subset(dat, core_grouping == "Other algae")
dim(other_algae_dat)

overall_model <- rma.mv(
    yi = yi, V = vi, mods = overall_model_formula,
    random = list(~ 1 | doi / ID), data = other_algae_dat, method = "REML"
)
summary(overall_model)

# Run in parallel for each core_grouping
core_groups <- unique(dat$core_grouping)
n_cores <- min(detectCores() - 1, length(core_groups))
results <- mclapply(core_groups, run_dredge_for_cg, dat = dat, overall_model_formula = overall_model_formula, mc.cores = n_cores)


out <- predict_surface(overall_model, dat, plot_anomalies = FALSE, mods_formula = overall_model_formula, backtransform = FALSE)


treatment_model_formula <- ~ dt_s + dph_s + dt_s:dph_s + factor(treatment) - 1
treatment_model <- rma.mv(
    yi = yi, V = vi, mods = treatment_model_formula,
    random = list(~1 | doi, ~1 | species_types, ~1 | ID), data = dat, method = "REML"
)
summary(treatment_model)


borrow_model_formula <- ~ dt_s + dph_s + dt_s:dph_s + factor(treatment) - 1
borrow_model <- rma.mv(
    yi = yi, V = vi, mods = borrow_model_formula,
    random = list(~1 | doi/ID, ~dt_s | core_grouping, ~dph_s | core_grouping), data = dat, method = "REML"
)
summary(borrow_model)



# -------------
# Exploring publication bias
# -------------
# calculate Rosenthal's fail-safe N
fsn(dat$yi, dat$vi)
# calculate Egger's regression test for funnel plot asymmetry
egger_test <- regtest(rma(yi = dat$yi, vi = dat$vi), model = "rma")

# save as png N.B. update file name and yaxis selection for seinv visulisation
png(
    "/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/calcification/analysis/rnative/funnel_plot.png",
    width = 12,
    height = 12,
    units = "in",
    res = 300
)
# Remove top 1% most extreme values for yi and vi
yi_threshold <- quantile(abs(dat$yi), 0.9, na.rm = TRUE)
vi_threshold <- quantile(abs(dat$vi), 0.9, na.rm = TRUE)
funnel_dat <- dat[abs(dat$yi) < yi_threshold & abs(dat$vi) < vi_threshold, ]
funnel(funnel_dat$yi, funnel_dat$vi,
    shade = c("white", "gray55", "gray75"),
    yaxs = "i", xaxs = "i",
    legend = TRUE, back = "gray90", hlines = NULL,
    xlab = "% Relative Calcification",
    ylab = "Standard Error",
    level = c(.1, .05, .01),
    las = 1, digits = list(1L, 0),
    # yaxis = "seinv",
    mgp = c(3, 1, 0),
    refline = 0,
)
dev.off()

# ------------
# Investigating effect of extreme values of effect sizes/variance
# ------------

extreme_filter_results <- extreme_filtering(dat, mods_formula, random_structure, plot = FALSE, extreme_limits = c(-1000, 1000), rescale_coefficients = TRUE)

png("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/calcification/analysis/rnative/myplot.png", width = 8, height = 8, units = "in", res = 300)
plot_extreme_filtering(extreme_filter_results$results, rescale_coefficients = TRUE)
dev.off()
# this shows that removing large variances doesn't change the results much, but removing extreme effect sizes does.
# it's reassuring that the 'large variance, may not obtain stable results' doesn't actually cause an issue




loo <- loo_analysis(dat_small, mods_formula, random_structure, level = "study")
p_combined <- plot_loo(loo, full_model = NULL)

# save/load to/from csv to avoid re-running
loo <- read.csv("loo_study_level.csv")
# write.csv(loo, file = "loo_study_level.csv", row.names = FALSE)
# these deviations show that some studies are very influential (even despite the large number of them). Don't know how to quantify which ones to leave out as a result: choosing to do this via Cook's distance instead.


# ------------
# Other influence metrics
# ------------
dat_small <- dat[1:400, ]

### For the entire dataset and each core_grouping, fit Cook's distance and plot along with the liberal and conservative thresholds
plot_cooks_distance_by_group(dat_small, mods_formula, random_structure, grouping_var = "core_grouping")
# TODO: marker types in 'all data' plot determined by core_grouping

# dat_small <- dat[1:200, ]
### Sensitivity analysis via influence filtering
sens <- influence_filtering(dat_small, mods_formula, random_structure, grouping_var = "core_grouping", plot = TRUE)
# TODO: where are the liberal threshold results for halimeda and other algae?


# ------------
# Check effect of extreme climate values
# ------------
# remove delta_ph and delta_t values below -0.4 and above 4

mods_formula = ~ dt_s + dph_s - 1
mods_formula = ~ dt + dph - 1
random_structure <- ~ 1 | doi / ID
clim_out <- clim_filtering(dat, mods_formula, random_structure, grouping_var = "core_grouping", dt_bounds = c(0, 4), dph_bounds = c(-0.4, 0), plot = TRUE, rescale_coefficients = TRUE)

png("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/calcification/analysis/rnative/myplot.png", width = 8, height = 10, units = "in", res = 300)

p <- plot_coefficient_comparison(clim_out$results)
print(p)
png("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/calcification/analysis/rnative/myplot.png", width = 8, height = 8, units = "in", res = 300)
print(p)
dev.off()


p <- plot_coefficient_comparison(out$results, return_plot=TRUE)

# ------------
# Check effect of original units
# ------------

unit_out <- unit_filtering(dat, mods_formula, random_structure, grouping_var = "st_calcification_unit", plot = TRUE, rescale_coefficients = FALSE)
p <- plot_coefficient_comparison(unit_out$results)
print(p)
png("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/calcification/analysis/rnative/myplot.png", width = 8, height = 8, units = "in", res = 300)
print(p)
dev.off()


# ------------
# Check effect of scaling moderators for fitting
# ------------


# Basic usage
scaling_results <- scaling_effects_test(
    dat = dat,
    mods_formula = ~ dt + dph - 1,
    random_structure = list(~1 | doi / ID, ~1 | species_types),
    title = "Raw vs Scaled Model Comparison"
)


raw_model <- rma.mv(
    yi = yi, V = vi, mods = ~ dt + dph - 1,
    random = list(~ 1 | doi / ID, ~ 1 | species_types), data = dat, method = "REML"
)

s_model <- rma.mv(
    yi = yi, V = vi, mods = ~ dt_s + dph_s - 1,
    random = list(~ 1 | doi / ID, ~ 1 | species_types), data = dat, method = "REML"
)

summary(raw_model)
summary(s_model)
sd_t <- sd(dat$dt, na.rm = TRUE)
sd_ph <- sd(dat$dph, na.rm = TRUE)


# Use the specialized plotting function for detailed comparison
p <- plot_scaling_comparison(scaling_results, show_differences = TRUE, width = 2, height = 2)
print(p)
png("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/calcification/analysis/rnative/myplot.png", width = 6, height = 6, units = "in", res = 300)
print(p)
dev.off()


# ------------
# Compare modified vs raw moderators
# ------------

coral_dat <- subset(dat, tolower(core_grouping) == "coral")
raw_mods_formula <- ~ dt + dph + I(dt^2) - 1
raw_mods_formula <- ~ dt + dph + dt:dph - 1
raw_mods_formula <- ~ dt + dph + dt:dph + I(dt^2) - 1
raw_mods_formula <- ~ dt + dph + dt:dph + I(dt^2) + I(dph^2) + I(dt^2):I(dph^2) - 1
raw_mods_formula <- ~ dt + dph - 1
fit_raw <- rma.mv(
    yi = yi, V = vi, mods = raw_mods_formula,
    random = random_structure, data = coral_dat, method = "REML"
)
summary(fit_raw)
multicollinearity_diagnostics(raw_mods_formula, coral_dat)
out <- predict_surface(fit_raw, coral_dat, plot_anomalies = TRUE, mods_formula = raw_mods_formula, backtransform = FALSE, fill_limits = c(-200, 0))


s_mods_formula <- ~ dt_s + dph_s + I(dt_s^2) - 1
s_mods_formula <- ~ dt_s + dph_s + dt_s:dph_s - 1
s_mods_formula <- ~ dt_s + dph_s + dt_s:dph_s + I(dt_s^2) - 1
s_mods_formula <- ~ dt_s + dph_s + dt_s:dph_s + I(dt_s^2) + I(dph_s^2) + I(dt_s^2):I(dph_s^2) - 1
s_mods_formula <- ~ dt_s + dph_s + dt_s:dph_s + I(dt_s^2) - 1
s_mods_formula <- ~ dt_s + dph_s - 1
fit_s <- rma.mv(
    yi = yi, V = vi, mods = s_mods_formula,
    random = random_structure, data = coral_dat, method = "REML"
)
summary(fit_s)
multicollinearity_diagnostics(s_mods_formula, coral_dat)
out <- predict_surface(fit_s, coral_dat, plot_anomalies = TRUE, mods_formula = s_mods_formula, fill_limits = c(-200, 0), backtransform = FALSE)

z_mods_formula <- ~ dt_z + dph_z + I(dt_z^2) - 1
z_mods_formula <- ~ dt_z + dph_z + dt_z:dph_z - 1
z_mods_formula <- ~ dt_z + dph_z + dt_z:dph_z + I(dt_z^2) - 1
z_mods_formula <- ~ dt_z + dph_z + dt_z:dph_z + I(dt_z^2) + I(dph_z^2) + I(dt_z^2):I(dph_z^2) - 1
z_mods_formula <- ~ dt_z + dph_z - 1
fit_z <- rma.mv(
    yi = yi, V = vi, mods = z_mods_formula,
    random = random_structure, data = coral_dat, method = "REML"
)
summary(fit_z)
multicollinearity_diagnostics(z_mods_formula, coral_dat)
out <- predict_surface(fit_z, coral_dat, plot_anomalies = TRUE, mods_formula = z_mods_formula,
fill_limits = c(-200, 0), backtransform = FALSE
)


cg_dat <- subset(dat, tolower(core_grouping) == "halimeda")
raw_mods_formula <- ~ dt + dph + I(dt^2) - 1
raw_mods_formula <- ~ dt + dph + dt:dph - 1
raw_mods_formula <- ~ dt + dph - 1
raw_mods_formula <- ~ dt + dph + dt:dph + I(dt^2) + I(dph^2) + I(dt^2):I(dph^2) - 1
raw_mods_formula <- ~ dt + dph + dt:dph + I(dt^2) - 1
fit_raw <- rma.mv(
    yi = yi, V = vi, mods = raw_mods_formula,
    random = random_structure, data = cg_dat, method = "REML"
)
summary(fit_raw)
multicollinearity_diagnostics(raw_mods_formula, cg_dat)
out <- predict_surface(fit_raw, cg_dat, plot_anomalies = TRUE, mods_formula = raw_mods_formula, fill_limits = c(-200, 0), backtransform = FALSE)

s_mods_formula <- ~ dt_s + dph_s + I(dt_s^2) - 1
s_mods_formula <- ~ dt_s + dph_s + dt_s:dph_s - 1
s_mods_formula <- ~ dt_s + dph_s + dt_s:dph_s + I(dt_s^2) + I(dph_s^2) + I(dt_s^2):I(dph_s^2) - 1
s_mods_formula <- ~ dt_s + dph_s + dt_s:dph_s + I(dt_s^2) - 1
s_mods_formula <- ~ dt_s + dph_s - 1
fit_s <- rma.mv(
    yi = yi, V = vi, mods = s_mods_formula,
    random = random_structure, data = cg_dat, method = "REML"
)
summary(fit_s)
multicollinearity_diagnostics(s_mods_formula, cg_dat)
out <- predict_surface(fit_s, cg_dat, plot_anomalies = TRUE, mods_formula = s_mods_formula, fill_limits = c(-200, 0), backtransform = FALSE)

z_mods_formula <- ~ dt_z + dph_z + I(dt_z^2) - 1
z_mods_formula <- ~ dt_z + dph_z + dt_z:dph_z - 1
z_mods_formula <- ~ dt_s + dph_z + dt_s:dph_z + I(dt_s^2) + I(dph_z^2) + I(dt_s^2):I(dph_z^2) - 1
z_mods_formula <- ~ dt_z + dph_z + dt_z:dph_z + I(dt_z^2) - 1
z_mods_formula <- ~ dt_z + dph_z - 1
fit_z <- rma.mv(
    yi = yi, V = vi, mods = z_mods_formula,
    random = random_structure, data = cg_dat, method = "REML"
)
summary(fit_z)
multicollinearity_diagnostics(z_mods_formula, cg_dat)
out <- predict_surface(fit_z, cg_dat, plot_anomalies = TRUE, mods_formula = z_mods_formula, fill_limits = c(-200, 0), backtransform = FALSE)


# -------------
# Core-grouping-wise dredge analysis
# -------------
# for each core grouping, fit a model with all combinations of moderators, then extract the top models (by AICc) and their weights
if (!requireNamespace("MuMIn", quietly = TRUE)) {
    install.packages("MuMIn")
}
library(MuMIn)
eval(metafor:::.MuMIn)


# For each core_grouping, fit all possible combinations of dt, dph, dt:dph, I(dt^2), I(dph^2), I(dt^2):I(dph^2) (no intercept)
# Use MuMIn::dredge to do model selection, extract top models by AICc and their weights

core_groups <- unique(tolower(dat$core_grouping))
dredge_results <- list()

fit_full <- rma.mv(
            yi = yi, V = vi, mods = dredge_formula,
            random = random_structure, data = subset(dat, tolower(core_grouping) == "cca"), method = "REML"
)
summary(fit_full)
# plot regression plot for dt
regplot(fit_full, mod = "dt", data = subset(dat, tolower(core_grouping) == "cca"), xlab = "delta_t", ylab = "yi")

for (cg in core_groups) {
    cat("\n--- Dredge for core_grouping:", cg, "---\n")
    dat_cg <- subset(dat, tolower(core_grouping) == cg)
    if (nrow(dat_cg) < 8) next # skip too-small groups

    # Full model formula (no intercept)
    dredge_formula <- ~ dt + dph + dt:dph + I(dt^2) + I(dph^2) + I(dt^2):I(dph^2) - 1

    # Fit the full model
    fit_full <- tryCatch(
        rma.mv(
            yi = yi, V = vi, mods = dredge_formula,
            random = random_structure, data = dat_cg, method = "REML"
        ),
        error = function(e) NULL
    )
    if (is.null(fit_full)) {
        cat("Model failed for group:", cg, "\n")
        next
    }

    # Dredge (all subsets)
    dd <- tryCatch(
        dredge(fit_full, trace = FALSE, rank = "AICc"),
        error = function(e) NULL
    )
    if (is.null(dd)) {
        cat("Dredge failed for group:", cg, "\n")
        next
    }

    # Extract top models (delta AICc < 2)
    top_models <- subset(dd, delta < 2)
    # Store results
    dredge_results[[cg]] <- list(
        dredge_table = as.data.frame(dd),
        top_models = as.data.frame(top_models)
    )

    cat("Top models for", cg, ":\n")
    print(top_models[, c("AICc", "delta", "weight")])
}

# save results to csv
if (!dir.exists("dredge_results")) dir.create("dredge_results")
for (cg in names(dredge_results)) {
    write.csv(dredge_results[[cg]]$dredge_table,
              file = sprintf("dredge_results/%s_dredge_table.csv", cg),
              row.names = FALSE)
}

# Optionally, inspect results:
# dredge_results[["coral"]]$top_models



# calculate the surfaces for each core grouping, then average them into a single prediction surface
# Calculate the prediction surface for each core_grouping using a common formula, then average into a single prediction
common_mods_formula <- ~ dt_z + dph_z + dt_z:dph_z - 1

# Get unique core_groupings (case-insensitive)
core_groups <- unique(tolower(dat$core_grouping))

# Store prediction surfaces for each group
surface_list <- list()
grid_list <- list()

for (cg in core_groups) {
    print(paste("Processing core_grouping:", cg))
    dat_cg <- subset(dat, tolower(core_grouping) == cg)
    if (nrow(dat_cg) < 5) next # skip too-small groups
    fit_cg <- rma.mv(
        yi = yi, V = vi, mods = common_mods_formula,
        random = random_structure, data = dat_cg, method = "REML"
    )
    # Get prediction surface (returns list with $grid and $pred)
    surface <- predict_surface(fit_cg, dat_cg, mods_formula = common_mods_formula, plot = FALSE, backtransform = FALSE)
    # Add group label to grid
    surface$grid$core_grouping <- cg
    surface_list[[cg]] <- surface$grid
    grid_list[[cg]] <- surface$grid
}

if (!requireNamespace("akima", quietly = TRUE)) {
    install.packages("akima")
}

library(akima)

# create single grid which covers the full range of dt and dph across all groups
all_dt <- unlist(lapply(grid_list, function(g) g$dt))
all_dph <- unlist(lapply(grid_list, function(g) g$dph))
dt_seq <- seq(min(all_dt, na.rm = TRUE), max(all_dt, na.rm = TRUE), length.out = 50)
dph_seq <- seq(min(all_dph, na.rm = TRUE), max(all_dph, na.rm = TRUE), length.out = 50)
common_grid <- expand.grid(dt = dt_seq, dph = dph_seq)
# Interpolate each group's predictions onto the common grid
interp_list <- list()
for (cg in names(surface_list)) {
    g <- surface_list[[cg]]
    interp_res <- with(g, akima::interp(
        x = dt, y = dph, z = pred,
        xo = dt_seq, yo = dph_seq, linear = TRUE, extrap = FALSE
    ))
    interp_df <- expand.grid(dt = interp_res$x, dph = interp_res$y)
    interp_df$pred <- as.vector(interp_res$z)
    interp_df$core_grouping <- cg
    interp_list[[cg]] <- interp_df
}

# Combine all interpolated data frames
combined_interp <- bind_rows(interp_list)
# Average predictions across core_groupings for each (dt, dph) pair
avg_surface <- combined_interp %>%
    group_by(dt, dph) %>%
    summarize(pred = mean(pred, na.rm = TRUE), .groups = "drop") %>%
    filter(!is.na(pred))

# plot the averaged surface using ggplot

# subtract the average value closest to (0,0) to center the plot around zero
center_value <- avg_surface %>%
    filter(abs(dt) == min(abs(dt)) & abs(dph) == min(abs(dph))) %>%
    pull(pred)
avg_surface$pred <- avg_surface$pred - center_value
# trim to shared range of dt and dph
shared_dt_range <- range(all_dt, na.rm = TRUE)
shared_dph_range <- range(all_dph, na.rm = TRUE)
avg_surface <- avg_surface %>%
    filter(dt >= shared_dt_range[1] & dt <= shared_dt_range[2]) %>%
    filter(dph >= shared_dph_range[1] & dph <= shared_dph_range[2])

# get the minimum range of dt and dph across all core_groupings
# Get the minimum shared range of dt and dph across all core_groupings
min_dt <- max(sapply(grid_list, function(g) min(g$dt, na.rm = TRUE)))
max_dt <- min(sapply(grid_list, function(g) max(g$dt, na.rm = TRUE)))
min_dph <- max(sapply(grid_list, function(g) min(g$dph, na.rm = TRUE)))
max_dph <- min(sapply(grid_list, function(g) max(g$dph, na.rm = TRUE)))

avg_surface_cropped <- avg_surface %>%
    filter(dt >= min_dt & dt <= max_dt) %>%
    filter(dph >= min_dph & dph <= max_dph)

# p <- ggplot(avg_surface, aes(x = dt, y = dph, fill = pred)) +
p <- ggplot(avg_surface_cropped, aes(x = dt, y = dph, fill = pred)) +
    geom_tile() +
    scale_fill_gradient2(
        low = "red",
        mid = "white",
        high = "blue",
        midpoint = 0,
        na.value = "transparent",
        name = "Predicted Effect Size"
    ) +
    labs(
        title = "Averaged Prediction Surface Across Core Groupings",
        x = "Delta Temperature (°C)",
        y = "Delta pH"
    ) +
    theme_minimal(base_size = 14) +
                labs(
                    x = expression(Delta ~ "Temperature"),
                    y = expression(Delta ~ "pH"),
                    title=NULL
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
                    panel.spacing = unit(0, "pt"),
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


#' Do the entire influence/refit/colinearity/contour plot for a given core_grouping value
fit_core_grouping <- function(dat, cg, mods_formula, random_structure, threshold_type = "conservative", remove_level = "sample", backtransform = TRUE) {
    # select data for this core_grouping
    if (!(cg %in% tolower(unique(dat$core_grouping)))) {
        stop(sprintf("%s must be a value in the 'core_grouping' column", cg))
    }
    dat_sub <- subset(dat, tolower(core_grouping) == tolower(cg))
    print(paste("Fitting for core_grouping =", cg, "with", nrow(dat_sub), "rows"))
    # Fit main model
    fit <- rma.mv(
        yi = yi, V = vi, mods = mods_formula,
        random = random_structure, data = dat_sub, method = "REML"
    )
    print("Initial fit:")
    print(summary(fit))
    # Influence diagnostics
    inf_measures <- compute_influence_measures(fit, dat_sub, parallel = "multicore", ncpus = 64)
    plot_influence_diagnostics(inf_measures, dat_sub, fit)
    influential_result <- identify_influential_studies(inf_measures, fit, dat_sub, method = threshold_type)
    refit_result <- refit_without_influential(dat_sub, influential_result, mods_formula, random_structure, remove_level = remove_level) # TODO: this isn't removing properly e.g. for halimeda
    fit_noinf <- refit_result$fit
    dat_noinf <- refit_result$data
    summary(fit_noinf)
    print("Refit without influential points:")
    print(summary(fit_noinf))
    # Multicollinearity
    multicollinearity_diagnostics(mods_formula, dat_sub)
    # Contour plot
    # grid <- predict_surface(fit, dat_sub, mods_formula)
    surface <- predict_surface(fit_noinf, dat_noinf, mods_formula)
    invisible(list(
        fit = fit,
        inf_measures = inf_measures,
        influential_result = influential_result,
        fit_noinf = fit_noinf,
        dat_noinf = dat_noinf,
        grid_noinf = surface$grid,
        pred_noinf = surface$pred
    ))
}

predict_surface(coral_res$fit_noinf, coral_res$dat_noinf, mods_formula, plot = TRUE)





mods_formula <- ~ dt + dph + dt:dph - 1
coral_res_simple_raw <- fit_core_grouping(dat, cg = "coral", mods_formula, random_structure)


mods_formula <- ~ dt_z + dph_z + dt_z:dph_z - 1
coral_res_simple_z <- fit_core_grouping(dat, cg = "coral", mods_formula, random_structure)

mods_formula <- ~ dt_z + dph_z + dt_z:dph_z
coral_res_simple_z_int <- fit_core_grouping(dat, cg = "coral", mods_formula, random_structure)

plot_surface_contour(coral_res_simple_z$grid_noinf, coral_res_simple_z$pred, plot_anomalies = TRUE)

plot_surface_contour(coral_res_simple_z_int$grid_noinf, coral_res_simple_z_int$pred, plot_anomalies = TRUE)



plot_regplot(coral_res_simple_z$fit_noinf, "dph_z", coral_res_simple_z$dat_noinf, "dph")

plot_regplot(coral_res_simple_raw$fit_noinf, "dph", coral_res_simple_raw$dat_noinf, "dph")



plot_regplot <- function(fit, mod, dat, colorby_col = "dph_z") {
    # Check if colorby_col exists in dat
    if (!(colorby_col %in% names(dat))) {
        stop(sprintf("Column '%s' not found in data.", colorby_col))
    }
    color_vals <- dat[[colorby_col]]
    # Remove NAs for coloring, but keep full data for regplot
    if (all(is.na(color_vals))) {
        stop(sprintf("All values in '%s' are NA.", colorby_col))
    }
    # If all values are the same, col_numeric will fail, so use a single color
    if (length(unique(na.omit(color_vals))) == 1) {
        col_fun <- function(x) rep(scales::viridis_pal()(1), length(x))
    } else {
        col_fun <- scales::col_numeric("viridis", domain = range(color_vals, na.rm = TRUE))
    }
    colors <- col_fun(color_vals)
    # If colors length doesn't match dat, pad with NA or recycle
    if (length(colors) != nrow(dat)) {
        colors <- rep(colors, length.out = nrow(dat))
    }
    regplot(
        fit,
        mod = mod,
        data = dat,
        col = colors,
        pch = 19
    )
    legend("topright",
        legend = c("Low delta_ph", "High delta_ph"),
        col = col_fun(range(color_vals, na.rm = TRUE)),
        pch = 19, title = "delta_ph"
    )
    abline(h = 0, col = "red", lty = 2)
    abline(v = 0, col = "red", lty = 2)
}


temp_exp <- subset(dat, treatment == "temp")
mods_formula <- ~ dt + I(dt^2)
coral_res_raw <- fit_core_grouping(temp_exp, cg = "coral", mods_formula, random_structure)

mods_formula <- ~ dt + dph + dt:dph
coral_res_noc <- fit_core_grouping(dat, cg = "coral", mods_formula, random_structure)

mods_formula <- ~ dt + dph + dt:dph
cca_res <- fit_core_grouping(dat, cg = "cca", mods_formula, random_structure)

mods_formula <- ~ dt + dph + dt:dph - 1
other_algae_res <- fit_core_grouping(dat, cg = "other algae", mods_formula, random_structure)
foraminifera_res <- fit_core_grouping(dat, cg = "foraminifera", mods_formula, random_structure)

mods_formula <- ~ dt + dph
halimeda_res <- fit_core_grouping(dat, cg = "halimeda", mods_formula, random_structure)

regplot(halimeda_res$fit_noinf, mod = "dt:dph", xlab = "delta_t", ylab = "yi")

cg_mods_formula <- ~ dt_z + dph_z + dt_z:dph_z


out <- fit_core_grouping(temp_exp, cg = "coral", cg_mods_formula, random_structure)
regplot(out$fit_noinf, mod = "dt_z", xlab = "delta_t", ylab = "yi")

predict_surface(out$fit_noinf, out$dat_noinf, cg_mods_formula, plot = TRUE)
# =============================
# Standardising and centering decreases collinearity which is apparently good for model convergence
# contour plots look sensible in terms of shape, but magnitudes are off, especially for some core groupings. Contour plotting is slightly broken (not robust to intercept/no intercept models)
# Have yet to test the leave-one-out sensitivity analysis or compare_models functions
# There are lots of influential studies

# -------------
# Influential studies investigation
# -------------

candidate_formulas <- list(
    ~1, # intercept-only baseline
    ~ dt + dph,
    ~ dt + dph + dt:dph,
    ~ dt + dph + I(dt^2) + I(dph^2),
    ~ dt + dph + I(dt^2) + I(dph^2) + dt:dph
)

# get influence for different model structures
investigate_formula_influences <- function(candidate_formulas, dat, random_structure) {
    results <- list()
    export_rows <- list()
    row_idx <- 1
    for (i in seq_along(candidate_formulas)) {
        f <- candidate_formulas[[i]]
        cat(sprintf("Fitting model %d/%d: %s\n", i, length(candidate_formulas), deparse(f)))
        fit <- tryCatch(
            {
                rma.mv(yi = yi, V = vi, mods = f, random = random_structure, data = dat, method = "REML")
            },
            error = function(e) {
                message("fit error for model ", i)
                return(NULL)
            }
        )
        if (is.null(fit)) next
        inf_measures <- compute_influence_measures(fit, dat, parallel = "multicore", ncpus = 64)
        results[[i]] <- list(
            formula = f,
            fit = fit,
            inf_measures = inf_measures,
            influential_result = list()
        )
        n_params <- length(coef(fit))
        n_studies <- length(unique(dat$doi))
        n_samples <- nrow(dat)
        # Add a summary row for this model (parameters and study count)
        export_rows[[row_idx]] <- data.frame(
            formula = deparse(f),
            method = "summary",
            threshold = NA,
            n_influential_studies = NA,
            n_influential_samples = NA,
            n_parameters = n_params,
            n_studies = n_studies,
            n_samples = n_samples,
            stringsAsFactors = FALSE
        )
        row_idx <- row_idx + 1

        for (method in c("conservative", "liberal")) {
            influential_result <- identify_influential_studies(inf_measures, fit, dat, method = method)
            results[[i]]$influential_result[[method]] <- influential_result

            # For each influential study, add a row to export_rows
            if (!is.null(influential_result$studies) && length(influential_result$studies) > 0) {
                for (study in influential_result$studies) {
                    export_rows[[row_idx]] <- data.frame(
                        formula = deparse(f),
                        method = method,
                        threshold = influential_result$threshold,
                        n_influential_studies = length(influential_result$studies),
                        n_influential_samples = dim(influential_result$samples)[1],
                        n_parameters = n_params,
                        n_studies = n_studies,
                        n_samples = n_samples,
                        stringsAsFactors = FALSE
                    )
                    row_idx <- row_idx + 1
                }
            } else {
                # If no influential studies, still record
                export_rows[[row_idx]] <- data.frame(
                    formula = deparse(f),
                    method = method,
                    threshold = influential_result$threshold,
                    n_influential_studies = 0,
                    n_influential_samples = 0,
                    n_parameters = n_params,
                    n_studies = n_studies,
                    n_samples = n_samples,
                    stringsAsFactors = FALSE
                )
                row_idx <- row_idx + 1
            }
        }
    }
    # Instead of exporting here, return the dataframe for user to export if desired
    export_df <- do.call(rbind, export_rows)
    return(list(
        results = results,
        export_df = export_df
    ))
}

# Accepts a named list (or named vector) of dataframes, returns influential studies for each
investigate_subgroup_influences <- function(candidate_formulas, sub_dats, random_structure) {
    results <- list()
    export_rows <- list()
    row_idx <- 1
    # sub_dats should be a named list: names(sub_dats) are subgroup names
    for (group in names(sub_dats)) {
        dat <- sub_dats[[group]]
        cat(sprintf("Investigating subgroup: %s\n", toupper(group)))
        res <- investigate_formula_influences(candidate_formulas, dat, random_structure)
        # For each formula, extract the influential studies result
        influential_studies <- lapply(res$results, function(x) {
            if (!is.null(x$influential_result)) {
                x$influential_result
            } else {
                NULL
            }
        })
        results[[group]] <- influential_studies

        # For export: collect all influential studies for this group
        export_df <- res$export_df
        if (!is.null(export_df) && nrow(export_df) > 0) {
            export_df$subgroup <- group
            export_rows[[row_idx]] <- export_df
            row_idx <- row_idx + 1
        }
    }
    # Instead of exporting here, return the dataframe for user to export if desired
    if (length(export_rows) > 0) {
        export_df_all <- do.call(rbind, export_rows)
        # Move 'subgroup' to first column for clarity
        export_df_all <- export_df_all[, c("subgroup", setdiff(names(export_df_all), "subgroup"))]
    } else {
        export_df_all <- data.frame()
    }
    return(list(
        results = results,
        export_df = export_df_all
    ))
}

# define a dictionary of subgroups: one key for each core grouping (and the entire dataset) with corresponding dataframes
sub_dats <- list(
    "other algae" = subset(dat, tolower(core_grouping) == "other algae"),
    "foraminifera" = subset(dat, tolower(core_grouping) == "foraminifera"),
    "halimeda" = subset(dat, tolower(core_grouping) == "halimeda"),
    "cca" = subset(dat, tolower(core_grouping) == "cca"),
    "coral" = subset(dat, tolower(core_grouping) == "coral"),
    "dat" = dat
)

res <- investigate_subgroup_influences(candidate_formulas, sub_dats, random_structure)

# To export to CSV for use in Python, run:
write.csv(res$export_df, file = "influential_studies_by_subgroup.csv", row.names = FALSE)


# -------------
# Trying general plot
# -------------

cg_fit <- rma.mv(
    yi = yi, V = vi, mods = ~ dt + dph + dt:dph + I(dt^2) + I(dph^2),
    random = ~ 1 | core_grouping / doi / ID, data = dat, method = "REML"
)
predict_surface(cg_fit, dat, ~ dt + dph + dt:dph + I(dt^2) + I(dph^2), plot = TRUE)

regplot(cg_fit, mod = "dt", xlab = "delta_t", ylab = "yi")


# -----------------------
# 1. Fit main (most complex) model
# -----------------------
complex_fit <- rma.mv(
    yi = yi, V = vi, mods = mods_formula,
    random = random_structure, data = dat, method = "REML"
)

summary(complex_fit)
regplot(complex_fit, mod = "dt", xlab = "delta_t", ylab = "yi")
regplot(complex_fit, mod = "dph", xlab = "delta_ph", ylab = "yi")
temp_exp <- subset(dat, treatment == "temp")
# remove rows for which dph is not zero
temp_exp <- temp_exp[temp_exp$dph == 0, ]
ph_exp <- subset(dat, treatment == "phtot")
plot(ph_exp$dph, ph_exp$yi)
plot(temp_exp$dt, temp_exp$yi)


mods_formula <- ~ dt + I(dt^2)
temp_fit <- rma.mv(
    yi = yi, V = vi, mods = mods_formula,
    random = random_structure, data = temp_exp, method = "REML"
)
summary(temp_fit)


xs <- seq(-1, 10, length = 500)
sav <- predict(temp_fit, newmods = cbind(xs, xs^2))
regplot(temp_fit,
    mod = 2, pred = sav, xvals = xs, las = 1, digits = 1, bty = "l",
    psize = .02 / sqrt(temp_exp$vi), xlab = "Predictor", main = "Quadratic Polynomial Model"
)


# -------------
# 2. Influence / leverage diagnostics
# -------------
inf_measures <- compute_influence_measures(complex_fit, dat, parallel = "multicore", ncpus = 64)
plot_influence_diagnostics(inf_measures, dat, complex_fit)

influential_result <- identify_influential_studies(inf_measures, complex_fit, dat, method = "conservative")
samples <- influential_result$samples
studies <- influential_result$studies
threshold <- influential_result$threshold
method <- influential_result$method

# --------------
# Refit model without influential studies
# --------------
refit_result <- refit_without_influential(dat, influential_result, mods_formula, random_structure, remove_level = "sample")
fit_noinf <- refit_result$fit
dat_noinf <- refit_result$data
summary(fit_noinf)

# -------------
# 3. Multicollinearity checks
# -------------
# Multicollinearity diagnostics function

# If VIFs >> 5 consider removing/recombining predictors.
multicollinearity_diagnostics(mods_formula, dat)

# -------------
# Extra: Sensitivity of slope estimates to point of interest
# -------------
# Sensitivity and prediction surface functions for metafor rma.mv models

# 1. Compute per-unit effect and SE for a z-scored model
per_unit_effect <- function(fit, dat, varname_z, varname_raw) {
    b <- coef(fit)
    V <- vcov(fit)
    sd_var <- sd(dat[[varname_raw]], na.rm = TRUE)
    beta_per_unit <- b[[varname_z]] / sd_var
    se_per_unit <- sqrt((1 / sd_var)^2 * V[varname_z, varname_z])
    list(beta = beta_per_unit, se = se_per_unit)
}


# -------------
# 4. Grouped k-fold cross-validation (group by doi)
# -------------
set.seed(42)
K <- 5
dois <- unique(dat$doi)
n_dois <- length(dois)
folds <- split(sample(dois), rep(1:K, length.out = n_dois))

cv_results <- data.frame(fold = 1:K, RMSE = NA, MAE = NA, Ntest = NA)

pb <- txtProgressBar(min = 0, max = K, style = 3)
for (k in seq_len(K)) {
    test_dois <- folds[[k]]
    train <- subset(dat, !(doi %in% test_dois))
    test <-se subset(dat, doi %in% test_dois)

    # fit on training set (must use the training vi's)
    fit_k <- tryCatch(
        rma.mv(yi = yi, V = vi, mods = mods_formula, random = random_structure, data = train, method = "REML"),
        error = function(e) {
            message("fit error on fold ", k)
            return(NULL)
        }
    )
    if (is.null(fit_k)) next

    # get fixed effect coefficients
    beta <- as.numeric(coef(fit_k))
    names(beta) <- names(coef(fit_k))

    # build model matrix for test set using same mods formula
    X_test <- model.matrix(as.formula(paste("~", deparse(mods_formula))), data = test)
    # ensure columns align: if any columns missing in test, add them as zeros
    need_cols <- setdiff(names(beta), colnames(X_test))
    if (length(need_cols) > 0) {
        X_test <- cbind(X_test, matrix(0,
            nrow = nrow(X_test), ncol = length(need_cols),
            dimnames = list(NULL, need_cols)
        ))
    }
    # reorder columns of X_test to match beta names
    X_test <- X_test[, names(beta), drop = FALSE]

    # linear predictor (fixed effects only)
    yhat <- as.vector(X_test %*% beta)

    # compute residuals/errors on observed yi
    resid <- test$yi - yhat
    RMSE <- sqrt(mean(resid^2, na.rm = TRUE))
    MAE <- mean(abs(resid), na.rm = TRUE)
    cv_results$RMSE[k] <- RMSE
    cv_results$MAE[k] <- MAE
    cv_results$Ntest[k] <- nrow(test)

    setTxtProgressBar(pb, k)
}
close(pb)

print(cv_results)
cat("Average RMSE:", mean(cv_results$RMSE, na.rm = TRUE), "Average MAE:", mean(cv_results$MAE, na.rm = TRUE), "\n")
#  TODO: explain this

# Note: this CV uses only fixed-effect prediction. If you want to include random-effect shrunken predictions
# you'd need to compute BLUPs for new clusters; but grouped CV by entire study avoids the need to predict new random effects.

# -------------
# 5. Bootstrap model-selection at study level
#    (resample studies, refit candidate models, collect best model per bootstrap)
# -------------
set.seed(42)
B <- 5 # number of bootstrap resamples, increase if you have compute power
candidate_names <- sapply(candidate_formulas, function(f) paste(deparse(f), collapse = ""))
sel_counts <- matrix(0,
    nrow = B, ncol = length(candidate_formulas),
    dimnames = list(NULL, candidate_names)
)

dois_all <- unique(dat$doi)
n_do <- length(dois_all)

pb <- txtProgressBar(min = 0, max = B, style = 3)
for (b in 1:B) {
    # sample studies with replacement
    sampled_dois <- sample(dois_all, size = n_do, replace = TRUE)
    dat_b <- do.call(rbind, lapply(sampled_dois, function(d) dat[dat$doi == d, , drop = FALSE]))

    aic_vals <- numeric(length(candidate_formulas))
    names(aic_vals) <- candidate_names

    # nested progress bar
    nested_pb <- txtProgressBar(min = 0, max = length(candidate_formulas), style = 3)
    on.exit(close(nested_pb), add = TRUE)
    for (i in seq_along(candidate_formulas)) {
        f <- candidate_formulas[[i]]
        # Try-fit; if fails, set large AIC
        fit_try <- tryCatch(
            {
                rma.mv(yi = yi, V = vi, mods = f, random = random_structure, data = dat_b, method = "REML")
            },
            error = function(e) NULL
        )
        if (is.null(fit_try)) {
            aic_vals[i] <- NA
        } else {
            # metafor has AIC() generic; use AIC for comparison
            aic_vals[i] <- AIC(fit_try)
        }
    }
    # choose the model with minimal AIC (ignoring NAs)
    if (all(is.na(aic_vals))) next
    best_i <- which.min(aic_vals)
    sel_counts[b, best_i] <- 1
}
close(pb)

sel_freq <- colSums(sel_counts, na.rm = TRUE) / B
sel_freq_sorted <- sort(sel_freq, decreasing = TRUE)
print(sel_freq_sorted)

# Show top few selected models and their selection proportions
cat("Bootstrap model selection frequencies:\n")
print(round(sel_freq_sorted, 3))

# -------------
# 6. AIC table for a given set of fitted models and deltaAIC / weights
# -------------
# Suppose you have a list of fitted models (fits_list)
# For demonstration, fit each candidate formula to the full data and gather AIC
fits_list <- list()
aic_list <- numeric(length(candidate_formulas))
for (i in seq_along(candidate_formulas)) {
    fits_list[[i]] <- tryCatch(
        {
            rma.mv(yi = yi, V = vi, mods = candidate_formulas[[i]], random = random_structure, data = dat, method = "REML")
        },
        error = function(e) NULL
    )
    aic_list[i] <- if (!is.null(fits_list[[i]])) AIC(fits_list[[i]]) else NA
}
aic_df <- data.frame(model = candidate_names, AIC = aic_list)
aic_df <- aic_df[order(aic_df$AIC), ]
aic_df$deltaAIC <- aic_df$AIC - min(aic_df$AIC, na.rm = TRUE)
aic_df$weight <- exp(-0.5 * aic_df$deltaAIC) / sum(exp(-0.5 * aic_df$deltaAIC), na.rm = TRUE)
print(aic_df)

# Guidance on interpreting deltaAIC:
# deltaAIC < 2 : essentially indistinguishable
# 2-7 : considerably less support
# >10 : essentially no support
# (these are rough rules of thumb; with very large AICs, compare differences not absolute numbers)

# -------------
# 7. Extra: Leave-one-study-out sensitivity (fast summary)
# -------------
study_ids <- unique(dat$doi)
nstud <- length(study_ids)
coef_mat <- matrix(NA, nrow = nstud, ncol = length(coef(complex_fit)))
colnames(coef_mat) <- names(coef(complex_fit))
rownames(coef_mat) <- study_ids

for (i in seq_along(study_ids)) {
    dat_sub <- subset(dat, doi != study_ids[i])
    fit_sub <- tryCatch(
        {
            rma.mv(yi = yi, V = vi, mods = mods_formula, random = random_structure, data = dat_sub, method = "REML")
        },
        error = function(e) NULL
    )
    if (!is.null(fit_sub)) coef_mat[i, ] <- coef(fit_sub)
}

# compute how much each coefficient changes vs full-data
full_coef <- coef(complex_fit)
delta_coef <- sweep(coef_mat, 2, full_coef, FUN = "-")
# e.g. max abs change for each parameter
max_abs_change <- apply(abs(delta_coef), 2, max, na.rm = TRUE)
print(max_abs_change)

# You can list studies where removal changes a coefficient by >X (tunable)
threshold <- 0.2 * abs(full_coef) # 20% change threshold
sensitive_flags <- apply(abs(delta_coef) > threshold, 2, any, na.rm = TRUE)
print("Parameters sensitive to study removal (20% change threshold):")
print(sensitive_flags)

# ==============================================================================
# INFLUENCE DIAGNOSTICS FUNCTIONS FOR META-ANALYSIS
# ==============================================================================

