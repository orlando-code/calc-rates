library(metafor)
library(ggplot2)

# Load the helper functions
source("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/calcification/analysis/rnative/helpers.R")

set.seed(123)

# -----------------------------
# Simulate realistic dummy data matching your structure
# -----------------------------
n_studies <- 100
obs_per_study <- 3
N <- n_studies * obs_per_study

# Create realistic moderators based on your actual data structure
dt <- rnorm(N, mean = 1.5, sd = 1.2) # temperature change, range roughly 0-4
dph <- rnorm(N, mean = -0.2, sd = 0.15) # pH change, range roughly -0.4 to 0
dph <- pmax(dph, -0.45) # ensure within realistic bounds
dt <- pmax(dt, -0.5) # ensure within realistic bounds

# Core groupings based on your actual taxonomy
core_grouping <- factor(sample(
    c("coral", "cca", "foraminifera", "halimeda", "other algae"),
    N,
    replace = TRUE,
    prob = c(0.4, 0.2, 0.15, 0.15, 0.1) # coral is most common
))

# Calcification units
st_calcification_unit <- factor(sample(
    c("μmol CaCO3 cm-2 h-1", "mg CaCO3 g-1 h-1", "% change", "mmol m-2 d-1"),
    N,
    replace = TRUE,
    prob = c(0.4, 0.3, 0.2, 0.1)
))

# Study identifiers
doi <- paste0("study_", rep(1:n_studies, each = obs_per_study))
ID <- 1:N

# Create realistic effect with strong climate trends
# Each taxonomic group responds differently to climate change
group_effects <- c(
    "coral" = -0.3, # corals strongly negatively affected
    "cca" = -0.2, # coralline algae moderately affected
    "foraminifera" = -0.1, # forams less affected
    "halimeda" = 0.05, # halimeda slightly positive
    "other algae" = 0.1 # other algae benefit
)

# Strong climate effects that should be detectable
beta0 <- 0.1 # baseline calcification rate
beta_dt <- -0.15 # strong negative effect of warming
beta_dph <- 0.8 # strong positive effect of pH (negative values = acidification harm)
beta_dt_dph <- -0.05 # interaction effect
beta_dt2 <- -0.02 # quadratic temperature effect (accelerating harm)

# Create the true response
mu <- beta0 +
    group_effects[core_grouping] +
    beta_dt * dt +
    beta_dph * dph +
    beta_dt_dph * dt * dph +
    beta_dt2 * dt^2

# Add study-level random effects (hierarchical structure)
study_re <- rnorm(n_studies, 0, 0.15)
mu <- mu + study_re[rep(1:n_studies, each = obs_per_study)]

# Add moderate group-level random effects within studies
group_re <- rnorm(N, 0, 0.1)
mu <- mu + group_re

# Observed effect sizes with realistic sampling variance
vi <- runif(N, 0.02, 0.25) # realistic range of sampling variances
yi <- rnorm(N, mu, sqrt(vi))

# Create scaled versions of moderators
dt_z <- scale(dt)[, 1] # z-score normalized
dph_z <- scale(dph)[, 1]
dt_s <- dt / sd(dt) # scaled by SD only
dph_s <- dph / sd(dph)

# Assemble the complete dataset
dat <- data.frame(
    yi = yi,
    vi = vi,
    dt = dt,
    dph = dph,
    dt_z = dt_z,
    dph_z = dph_z,
    dt_s = dt_s,
    dph_s = dph_s,
    core_grouping = core_grouping,
    st_calcification_unit = st_calcification_unit,
    doi = doi,
    ID = ID,
    stringsAsFactors = FALSE
)

cat("Dataset created with", nrow(dat), "observations\n")
cat("Core groupings:", table(dat$core_grouping), "\n")
cat("DT range:", round(range(dat$dt), 2), "\n")
cat("DPH range:", round(range(dat$dph), 3), "\n")

# TODO: plot the true response as a contour plot

# -----------------------------
# Define random structure matching your real analysis
# -----------------------------
random_structure <- ~ 1 | doi / ID

# -----------------------------
# Test scaling effects with realistic models
# -----------------------------

# Model 1: Raw moderators with interaction
mods_formula_raw <- ~ dt + dph + dt:dph

# Model 2: Z-score normalized moderators
mods_formula_z <- ~ dt_z + dph_z + dt_z:dph_z

# Model 3: SD-scaled moderators
mods_formula_s <- ~ dt_s + dph_s + dt_s:dph_s

cat("\n=== FITTING MODELS ===\n")

# Fit all three models
fit_raw <- rma.mv(
    yi = yi, V = vi,
    mods = mods_formula_raw,
    random = random_structure,
    data = dat, method = "REML"
)

fit_z <- rma.mv(
    yi = yi, V = vi,
    mods = mods_formula_z,
    random = random_structure,
    data = dat, method = "REML"
)

fit_s <- rma.mv(
    yi = yi, V = vi,
    mods = mods_formula_s,
    random = random_structure,
    data = dat, method = "REML"
)

cat("Raw model coefficients:\n")
print(coef(fit_raw))

cat("\nZ-score model coefficients:\n")
print(coef(fit_z))

cat("\nSD-scaled model coefficients:\n")
print(coef(fit_s))

# -----------------------------
# Compare predictions using contour plots
# -----------------------------
cat("\n=== GENERATING PREDICTION SURFACES ===\n")

# Raw model predictions
cat("Generating raw model predictions...\n")
surface_raw <- predict_surface(
    fit_raw, dat, mods_formula_raw,
    backtransform = FALSE,
    plot = TRUE,
    title = "Raw Moderators (dt, dph)",
    plot_anomalies = TRUE
)

# Z-score model predictions
cat("Generating z-score model predictions...\n")
surface_z <- predict_surface(
    fit_z, dat, mods_formula_z,
    backtransform = FALSE,
    plot = TRUE,
    title = "Z-score Normalized (dt_z, dph_z)",
    plot_anomalies = TRUE
)

# SD-scaled model predictions
cat("Generating SD-scaled model predictions...\n")
surface_s <- predict_surface(
    fit_s, dat, mods_formula_s,
    backtransform = FALSE,
    plot = TRUE,
    title = "SD-scaled (dt_s, dph_s)",
    plot_anomalies = TRUE
)

# plot all the surfaces on the same plot
ggplot() +
    geom_contour(data = surface_raw$grid, aes(x = dt, y = dph, z = pred), color = "red") +
    geom_contour(data = surface_z$grid, aes(x = dt_z, y = dph_z, z = pred), color = "blue") +
    geom_contour(data = surface_s$grid, aes(x = dt_s, y = dph_s, z = pred), color = "green") +
    theme_minimal()

# -----------------------------
# Direct comparison at specific points
# -----------------------------
cat("\n=== POINT PREDICTIONS COMPARISON ===\n")

# Create test points for comparison
test_points <- data.frame(
    dt = c(0, 1, 2, 3),
    dph = c(0, -0.1, -0.2, -0.3)
)

# Add scaled versions
test_points$dt_z <- (test_points$dt - mean(dat$dt)) / sd(dat$dt)
test_points$dph_z <- (test_points$dph - mean(dat$dph)) / sd(dat$dph)
test_points$dt_s <- test_points$dt / sd(dat$dt)
test_points$dph_s <- test_points$dph / sd(dat$dph)

# Generate predictions
pred_raw <- predict(fit_raw, newmods = model.matrix(mods_formula_raw, data = test_points))
pred_z <- predict(fit_z, newmods = model.matrix(mods_formula_z, data = test_points))
pred_s <- predict(fit_s, newmods = model.matrix(mods_formula_s, data = test_points))

comparison_df <- data.frame(
    dt = test_points$dt,
    dph = test_points$dph,
    pred_raw = pred_raw$pred,
    pred_z = pred_z$pred,
    pred_s = pred_s$pred,
    diff_raw_z = pred_raw$pred - pred_z$pred,
    diff_raw_s = pred_raw$pred - pred_s$pred
)

cat("Prediction comparison at key points:\n")
print(round(comparison_df, 4))

cat(
    "\nMaximum absolute difference (raw vs z-score):",
    round(max(abs(comparison_df$diff_raw_z)), 4), "\n"
)
cat(
    "Maximum absolute difference (raw vs SD-scaled):",
    round(max(abs(comparison_df$diff_raw_s)), 4), "\n"
)

# -----------------------------
# Summary statistics
# -----------------------------
cat("\n=== MODEL COMPARISON SUMMARY ===\n")
cat(
    "Raw model - AIC:", round(AIC(fit_raw), 2),
    "| QE:", round(fit_raw$QE, 2),
    "| QM:", round(fit_raw$QM, 2), "\n"
)
cat(
    "Z-score model - AIC:", round(AIC(fit_z), 2),
    "| QE:", round(fit_z$QE, 2),
    "| QM:", round(fit_z$QM, 2), "\n"
)
cat(
    "SD-scaled model - AIC:", round(AIC(fit_s), 2),
    "| QE:", round(fit_s$QE, 2),
    "| QM:", round(fit_s$QM, 2), "\n"
)

# Store results for further analysis
scaling_test_results <- list(
    data = dat,
    models = list(raw = fit_raw, z = fit_z, s = fit_s),
    surfaces = list(raw = surface_raw, z = surface_z, s = surface_s),
    comparison = comparison_df
)
