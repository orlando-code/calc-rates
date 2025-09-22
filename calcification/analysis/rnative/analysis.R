# Load required libraries
library(metafor)
library(dplyr)
library(ggplot2)
library(knitr)
library(corrplot)


install.packages("languageserver")
install.packages('httpgd', repos = c('https://community.r-multiverse.org', 'https://cloud.r-project.org'))

# Load data
# Use a relative path to keep things neater
data_df <- read.csv("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/data/clean/analysis_ready_data.csv")
response_var <- "st_relative_calcification"
response_var_var <- paste(response_var, "var", sep = "_")
# remove rows with NA in response_var or response_var_var
data_df <- data_df[!is.na(data_df[[response_var]]) & !is.na(data_df[[response_var_var]]), ]
# remove extreme variances
data_df <- data_df[data_df[[response_var_var]] < 5000, ]

# select only rows for which st_calcification_unit is equal to mgCaCO3 g-1d-1
cat("Unique values in st_calcification_unit:\n")
print(unique(data_df$st_calcification_unit))
# Adjust the filter below to match the actual value exactly as it appears in your data
data_df <- data_df %>% filter(
  # st_calcification_unit == "mgCaCO3 g-1d-1", 
  treatment == "temp" | treatment == "phtot" | treatment == "temp_phtot"
)

# get top n rows. 360-370 nullifies results with no core grouping term
test_df <- data_df %>% slice_head(n = 360)
# select rows 362 to 370
# test_df <- data_df %>% slice(362:370)
test_df <- data_df

# check for nans
cat("Number of NAs in original_doi:", sum(is.na(test_df$original_doi)), "\n")
cat("Number of NAs in species_types:", sum(is.na(test_df$species_types)), "\n")
cat("Number of NAs in delta_t:", sum(is.na(test_df$delta_t)), "\n")
cat("Number of NAs in delta_ph:", sum(is.na(test_df$delta_ph)), "\n")


# --- check data ---
# Display basic information
cat("Dataset dimensions:", dim(test_df), "\n")
cat("Number of study locations:", length(unique(test_df$doi)), "\n")
cat("Number of effect sizes:", nrow(test_df), "\n")

# Display column names
cat("Columns in dataset:\n")
print(names(test_df))

# --- run simplest random effects metafor model ---
model <- rma(yi = test_df[[response_var]], vi = test_df[[response_var_var]], data = test_df)
print(summary(model))


# plot influence
cd <- cooks.distance.rma.mv(t_model, progbar=TRUE,
        reestimate=FALSE, parallel="multicore", ncpus=64, cl=NULL)
# export to csv
write.csv(cd, file = "cd_r_no_reestimate.csv")

plot(cd, type = "o", pch = 19, xlab = "Observed Outcome", ylab = "Cook's Distance")
threshold <- 2 * sqrt((8 / (dim(test_df)[1] - 8 - 1)))
plot(cd, type = "o", pch = 19, xlab = "Observed Outcome", ylab = "Cook's Distance")
abline(h = threshold, col = "red", lty = 2)
# remove points above threshold and re-plot
test_df <- test_df[cd < threshold, ]

# --- add a moderator variable ---
t_model <- rma.mv(
  yi = test_df[[response_var]],
  V = test_df[[response_var_var]],
  data = test_df,
  mods = ~ delta_t * delta_ph + I(delta_t^2) * I(delta_ph^2) + factor(core_grouping) - 1,
  random = ~ 1 | original_doi / species_types,
)
print(summary(t_model))
# plot regplot
regplot(t_model, mod = 2, xlab = "Temperature", ylab = "Relative calcification",
# xlim=c(0,10)
)

# get number of nans in original_doi and species_types
cat("Number of NAs in original_doi:", sum(is.na(test_df$original_doi)), "\n")
cat("Number of NAs in species_types:", sum(is.na(test_df$species_types)), "\n")

# remove rows with NA in original_doi or species_types
test_df <- test_df[!is.na(test_df$original_doi) & !is.na(test_df$species_types), ]  

new_t_model <- rma.mv(
  yi = test_df[[response_var]],
  V = test_df[[response_var_var]],
  data = test_df,
  mods = ~ delta_t*delta_ph + I(delta_t^2)*I(delta_ph^2),
  random = ~ 1| original_doi/species_types,
  # rho = 0.01 # This has no effect in this model structure
)
print(summary(new_t_model))
# plot regplot
regplot(new_t_model, xlab = "Temperature", ylab = "Relative calcification", )

# --- add a moderator variable ---
# remove rows with delta_t < 1
# test_df_dt <- test_df[test_df$delta_t >= 1, ]
# test_df_df <- test_df
# Print model summary
dt_1_model <- rma(yi = test_df_dt[[response_var]], vi = test_df_dt[[response_var_var]], data = test_df_dt, mods = ~ delta_t)
print(summary(dt_1_model))
# plot regplot
regplot(dt_1_model, xlab = expression(paste(Delta, "Temperature")), ylab = response_var, 
# ylim=c(-200,200)
)

# fit a quadratic model
dt_2_model <- rma(yi = test_df[[response_var]], vi = test_df[[response_var_var]], data = test_df, mods = ~ delta_t + I(delta_t^2))
print(summary(dt_2_model))
# plot nonlinear fit to data as estimated by the model
xs <- seq(0, 10, length=500)
sav <- predict(dt_2_model, newmods=unname(poly(xs, degree=2, raw=TRUE)))
regplot(dt_2_model, mod=2, pred=sav, xvals=xs, las=1, digits=1, bty="l",
        psize=10/sqrt(test_df[[response_var_var]]), xlab="Predictor", main="Quadratic Polynomial Model",
        ylim=c(-100,100)
        )

# --- leave1out sensitivity analysis ---

# optionally subset the data to a smaller number of studies e.g. for demonstration/debugging
subset_n_studies <- NA  # Set to a number (e.g., 10) to subset, or NA to use all studies

if (!is.na(subset_n_studies)) {
  selected_studies <- unique(test_df$original_doi)[1:subset_n_studies]
  test_df <- test_df[test_df$original_doi %in% selected_studies, ]
  cat("Subsetting to", subset_n_studies, "studies. New dataset dimensions:", dim(test_df), "\n")
} else {
  cat("Using all studies. Dataset dimensions:", dim(test_df), "\n")
}

model <- rma(yi = test_df[[response_var]], vi = test_df[[response_var_var]], data = test_df)

# perform leave-one-study-out analysis, clustering by original_doi (i.e., leaving out one study at a time)
leave1out_res <- leave1out(model, cluster = test_df$original_doi, progbar = TRUE)
# visualise the results