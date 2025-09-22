# Required package
library(metafor)
library(dplyr)
library(parallel)
if (!requireNamespace("meta", quietly = TRUE)) {
    install.packages("meta")
}
library(meta)


# ----------------------------
# 0. Prep / user settings
# ----------------------------
# data frame: dat
# required columns in dat: yi, vi, doi, <moderators...>
dat <- read.csv("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/calc-rates/data/clean/analysis_ready_data.csv")
# remove rows with NA in yi or vi
dat <- dat[!is.na(dat$st_relative_calcification) & !is.na(dat$st_relative_calcification_var), ]
# remove extreme variances
dat <- dat[dat$st_relative_calcification_var < 5000, ]
# rename for convenience
colnames(dat)[which(colnames(dat) == "st_relative_calcification")] <- "yi"
colnames(dat)[which(colnames(dat) == "st_relative_calcification_var")] <- "vi"

# centre only
dat$dt <- dat$delta_t
dat$dph <- dat$delta_ph

dat$dt_c <- dat$delta_t - mean(dat$delta_t, na.rm = TRUE)
dat$dph_c <- dat$delta_ph - mean(dat$delta_ph, na.rm = TRUE)

# scale only
dat$dt_s <- dat$delta_t / sd(dat$delta_t, na.rm = TRUE)
dat$dph_s <- dat$delta_ph / sd(dat$delta_ph, na.rm = TRUE)

dat$dt_z <- (dat$delta_t - mean(dat$delta_t, na.rm = TRUE)) / sd(dat$delta_t, na.rm = TRUE)
dat$dph_z <- (dat$delta_ph - mean(dat$delta_ph, na.rm = TRUE)) / sd(dat$delta_ph, na.rm = TRUE)


m.gen <- metagen(
    TE = yi,
    seTE = vi,
    studlab = ID,
    data = dat,
    sm = "SMD",
    fixed = FALSE,
    random = TRUE,
    method.tau = "REML",
    method.random.ci = "HK",
    title = "Calc rates"
)

update(m.gen,
    subgroup = core_grouping,
    tau.common = FALSE
)
update(m.gen,
    # subgroup = st_calcification_unit, # Fisher scoring algorithm did not converge.
    subgroup = treatment, # Fisher scoring algorithm did not converge.
    tau.common = FALSE
)
