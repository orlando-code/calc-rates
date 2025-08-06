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
# select only rows for which st_calcification_unit is equal to mgCaCO3 g-1d-1
cat("Unique values in st_calcification_unit:\n")
print(unique(data_df$st_calcification_unit))
# Adjust the filter below to match the actual value exactly as it appears in your data
data_df <- data_df %>% filter(st_calcification_unit == "mgCaCO3 g-1d-1", 
treatment == "temp")

# --- check data ---
# Display basic information
cat("Dataset dimensions:", dim(data_df), "\n")
cat("Number of study locations:", length(unique(data_df$doi)), "\n")
cat("Number of effect sizes:", nrow(data_df), "\n")

# Display column names
cat("Columns in dataset:\n")
print(names(data_df))

# --- run simplest random effects metafor model ---
model <- rma(yi = data_df[[response_var]], vi = data_df[[response_var_var]], data = data_df)
print(summary(model))


# --- add a moderator variable ---
# print model summary
t_model <- rma(yi = data_df[[response_var]], vi = data_df[[response_var_var]], data = data_df, mods = ~ temp)
print(summary(t_model))
# plot regplot
regplot(t_model, xlab = "Temperature", ylab = "Relative calcification", )

# --- add a moderator variable ---
# remove rows with delta_t < 1
# data_df_dt <- data_df[data_df$delta_t >= 1, ]
# data_df_df <- data_df
# Print model summary
dt_1_model <- rma(yi = data_df_dt[[response_var]], vi = data_df_dt[[response_var_var]], data = data_df_dt, mods = ~ delta_t)
print(summary(dt_1_model))
# plot regplot
regplot(dt_1_model, xlab = expression(paste(Delta, "Temperature")), ylab = response_var, 
# ylim=c(-200,200)
)

# fit a quadratic model
dt_2_model <- rma(yi = data_df[[response_var]], vi = data_df[[response_var_var]], data = data_df, mods = ~ delta_t + I(delta_t^2))
print(summary(dt_2_model))
# plot nonlinear fit to data as estimated by the model
xs <- seq(0, 10, length=500)
sav <- predict(dt_2_model, newmods=unname(poly(xs, degree=2, raw=TRUE)))
regplot(dt_2_model, mod=2, pred=sav, xvals=xs, las=1, digits=1, bty="l",
        psize=10/sqrt(data_df[[response_var_var]]), xlab="Predictor", main="Quadratic Polynomial Model",
        ylim=c(-100,100)
        )

# --- leave1out sensitivity analysis ---

# optionally subset the data to a smaller number of studies e.g. for demonstration/debugging
subset_n_studies <- NA  # Set to a number (e.g., 10) to subset, or NA to use all studies

if (!is.na(subset_n_studies)) {
  selected_studies <- unique(data_df$original_doi)[1:subset_n_studies]
  data_df <- data_df[data_df$original_doi %in% selected_studies, ]
  cat("Subsetting to", subset_n_studies, "studies. New dataset dimensions:", dim(data_df), "\n")
} else {
  cat("Using all studies. Dataset dimensions:", dim(data_df), "\n")
}

model <- rma(yi = data_df[[response_var]], vi = data_df[[response_var_var]], data = data_df)

# perform leave-one-study-out analysis, clustering by original_doi (i.e., leaving out one study at a time)
leave1out_res <- leave1out(model, cluster = data_df$original_doi, progbar = TRUE)
# visualise the results
