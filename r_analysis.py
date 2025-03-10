import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import StrVector, FloatVector, IntVector

# Enable automatic conversion between pandas and R dataframes
pandas2ri.activate()

# Function to ensure R packages are installed
def ensure_r_packages():
    # Get R's utils package
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list

    # List of required packages
    required_packages = ["metafor", "dplyr", "tidyr", "MASS", "ggplot2", 
                         "viridis", "patchwork", "magick", "ggsci", "ggpubr"]
    
    # Get list of installed packages
    installed_packages = ro.r('rownames(installed.packages())')
    
    # Install missing packages
    for pkg in required_packages:
        if pkg not in installed_packages:
            print(f"Installing R package: {pkg}")
            utils.install_packages(pkg)
    
    print("All required R packages are installed.")

# Run the package installation check
ensure_r_packages()

# Now import the packages
base = importr('base')
try:
    metafor = importr('metafor')
    dplyr = importr('dplyr')
    tidyr = importr('tidyr')
    MASS = importr('MASS')
    ggplot2 = importr('ggplot2')
    parallel = importr('parallel')
except Exception as e:
    print(f"Error importing R packages: {e}")
    print("You might need to restart your Python session after installing packages.")
    raise

# Function to set up R environment
def setup_r_environment():
    # Install required packages if needed
    ro.r('''
    required_packages <- c("metafor", "dplyr", "tidyr", "MASS", "ggplot2", "viridis", 
                           "patchwork", "magick", "ggsci", "ggpubr", "parallel")
    
    for(package in required_packages) {
        if(!require(package, character.only = TRUE)) {
            install.packages(package)
            library(package, character.only = TRUE)
        }
    }
    ''')
    
    # Define custom ggplot theme
    ro.r('''
    theme_cca <- function() {
      theme_classic(base_size = 12) +
        theme(
          panel.border = element_rect(colour = "black", fill = NA, linetype = 1),
          panel.background = element_rect(fill = "transparent"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.line = element_blank(),
          plot.title = element_text(face = "italic"),
          axis.text = element_text(colour = "black"),
          axis.ticks = element_line(colour = "black"),
          legend.position = c(0.1,0.925),
          legend.text.align = 0, 
          legend.background = element_blank(), 
          legend.spacing = unit(0, "lines"), 
          legend.key = element_blank(), 
          legend.spacing.y = unit(0.25, 'cm'),
          legend.title = element_blank()
        ) 
    }
    
    # Define color scheme
    colour.scheme <- c("black",
                       pal_npg(palette = c("nrc"), alpha = 1)(10)[4],
                       pal_npg(palette = c("nrc"), alpha = 0.6)(10)[10],
                       pal_npg(palette = c("nrc"), alpha = 0.6)(10)[1]
    )
    ''')
    
    print("R environment setup complete")

# Function to load and prepare adult data
def prepare_adult_data(file_path):
    # Read data in Python
    adult_data = pd.read_csv(file_path)
    
    # Initial filtering in Python
    adult_data = adult_data[adult_data['Family'] != ""]
    adult_data = adult_data.dropna(subset=['dpHt'])
    adult_data['id'] = adult_data.index.astype(str)
    
    # Calculate Hedges G correction
    adult_data['correction'] = 1 - (3/(4*(adult_data['n']*2)-9))
    adult_data['corrected_hedges'] = adult_data['Hedges G'] * adult_data['correction']
    adult_data['corrected_variance'] = adult_data['Variance'] * (adult_data['correction']**2)
    
    # Convert to R dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_adult_data = ro.conversion.py2rpy(adult_data)
    
    # Store in R environment as adult.data.hedges for compatibility with original R code
    ro.r.assign('adult.data.hedges', r_adult_data)
    
    # Return both Python and R versions
    return adult_data, r_adult_data

# Function to check for outliers
def check_for_outliers(r_adult_data, ncpus=8):
    # Run random-effects multi-level model
    model_formula = ro.r('~ dpHt + Temperature + `Irradiance (umol photons m-2 s-1)` + `Duration (days)` + factor(Family) - 1')
    
    ro.r.assign('adult_data', r_adult_data)
    model = ro.r('''
    model.full.adult.hedges.id.nested <- rma.mv(yi = corrected_hedges, 
                  V = corrected_variance, 
                  data = adult_data, 
                  method = "REML", 
                  mods = ~ dpHt + Temperature + `Irradiance (umol photons m-2 s-1)` + `Duration (days)` + factor(Family) - 1,
                  test="z",
                  random = list(~ 1| (id/Study)))
    model.full.adult.hedges.id.nested
    ''')
    
    # Calculate Cook's distance
    cooks_distance = ro.r(f'''
    model.full.adult.hedges.cooks <- 
    cooks.distance.rma.mv(model.full.adult.hedges.id.nested, progbar=TRUE,
            reestimate=TRUE, parallel="multicore", ncpus={ncpus}, cl=NULL)
    model.full.adult.hedges.cooks
    ''')
    
    # Calculate threshold
    n = len(r_adult_data)
    k = ro.r('length(coef(model.full.adult.hedges.id.nested))')[0]
    threshold = 2 * np.sqrt((k+1)/(n-k-1))
    
    # Identify outliers
    with localconverter(ro.default_converter + pandas2ri.converter):
        cooks_df = ro.conversion.rpy2py(ro.r('''
        as.data.frame(model.full.adult.hedges.cooks)
        '''))
    
    cooks_df = pd.DataFrame({'value': cooks_df.iloc[:, 0]})
    cooks_df['Study'] = range(1, len(cooks_df) + 1)
    outliers = cooks_df[cooks_df['value'] > threshold]
    
    # Create plot in R
    ro.r('''
    plot(model.full.adult.hedges.cooks, type="o", pch=19, xlab="Study", ylab="Cook's Distance", xaxt="n")
    axis(side=1, at=seq_along(model.full.adult.hedges.cooks), labels=as.numeric(names(model.full.adult.hedges.cooks)))
    ''')
    
    return outliers

# Function to run multi-level meta-analysis
def run_meta_analysis_model():
    # Run the multi-level meta-analysis model
    ro.r("""
    # Random-effects multi-level model for assessing calcification in adults
    model.full.adult.hedges.id.nested <- rma.mv(yi = corrected_hedges, 
                  V = corrected_variance, 
                  data = adult.data.hedges, 
                  method = "REML", 
                  mods = ~ dpHt + Temperature + `Irradiance (umol photons m-2 s-1)` + `Duration (days)` + factor(Family) - 1,
                  test="z",
                  random = list(~ 1| (id/Study)))
    
    # Print summary to console
    print(summary(model.full.adult.hedges.id.nested))
    """)
    
    print("Multi-level meta-analysis model complete")

# Function to create overall plot
def create_overall_plot(output_dir="output"):
    # First create the plot
    
    # Access and print information about the model structure
    print("Model moderators:")
    mod_names = ro.r("names(coef(model.full.adult.hedges.id.nested))")
    print(mod_names)
    
    # Extract the exact name of the dpHt moderator from the model directly in R
    ro.r("""
    # Find the moderator name containing dpHt
    mod_names <- names(coef(model.full.adult.hedges.id.nested))
    dpHt_mod <- mod_names[grep("dpHt", mod_names)][1]
    
    # Overall plot of all studies assessing calcification in adults
    svg(file = 'output/adult_overall_bubble_with_id_nested.svg', width = 8, height = 8) 
    regplot.rma(model.full.adult.hedges.id.nested, mod = dpHt_mod, refline = 0, 
                xlim = c(-1,0.25), ylim = c(-15,15), predlim = c(-1,0.25),
                ylab = "Hedges' g", xlab = expression(paste("\\U0394", "pH")[T]))
    dev.off()
    """.replace('output', output_dir))
    
    print(f"Overall plot saved to {output_dir}/adult_overall_bubble_with_id_nested.svg")

# Function to create family-specific plots
def create_family_plots(output_dir="output", families=None):
    if families is None:
        # Extract unique families from the dataset
        families = ro.r('unique(adult.data.hedges$Family)')
    
    # Create a model and plot for each family
    for family in families:
        family_name = str(family)
        
        # Skip empty family names
        if not family_name.strip():
            continue
            
        print(f"Creating plot for family: {family_name}")
        
        # Check if there are enough studies for this family to run a meta-analysis
        study_count = ro.r(f'sum(adult.data.hedges$Family == "{family_name}")')[0]
        
        if study_count <= 1:
            print(f"Skipping {family_name} because it has only {study_count} study/studies")
            continue
        
        try:
            ro.r(f'''
            # Try to run the model for {family_name}
            tryCatch({{
                # Model for {family_name}
                model.full.adult.hedges.id.nested.{family_name} <- rma.mv(yi = corrected_hedges, 
                          V = corrected_variance, 
                          data = adult.data.hedges, 
                          method = "REML", 
                          mods = ~ dpHt + Temperature - 1,
                          test="z",
                          subset = (Family == "{family_name}"),
                          random = list(~ 1| id/Study))
                
                # Print model summary
                print(summary(model.full.adult.hedges.id.nested.{family_name}))
                
                # Create SVG plot
                svg(file = '{output_dir}/adult_{family_name}_bubble_with_id_nested.svg', width = 8, height = 8) 
                regplot.rma(model.full.adult.hedges.id.nested.{family_name}, mod = "dpHt", refline = 0, 
                            xlim = c(-1,0.25), ylim = c(-15,15), predlim = c(-1,0.25),
                            ylab = "Hedges' g", xlab = expression(paste("\\U0394", "pH")[T]))
                dev.off()
            }}, error = function(e) {{
                print(paste("Error running model for family {family_name}:", e$message))
            }})
            ''')
        except Exception as e:
            print(f"Python exception when processing family {family_name}: {e}")

# Main function to run the complete meta-analysis pipeline
def run_complete_meta_analysis(file_path, output_dir="output", ncpus=8):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup R environment and define custom functions
    setup_r_environment()
    
    # Prepare adult data
    print("Preparing adult data...")
    adult_data_py, adult_data_r = prepare_adult_data(file_path)
    
    # Check for outliers
    print("Checking for outliers...")
    outliers = check_for_outliers(adult_data_r, ncpus)
    
    # Save outlier plot
    ro.r(f'pdf("{output_dir}/cooks_distance_plot.pdf")')
    ro.r('''
    plot(model.full.adult.hedges.cooks, type="o", pch=19, xlab="Study", ylab="Cook's Distance", xaxt="n")
    axis(side=1, at=seq_along(model.full.adult.hedges.cooks), labels=as.numeric(names(model.full.adult.hedges.cooks)))
    ''')
    ro.r('dev.off()')
    
    # Remove outliers from dataset
    print(f"Found {len(outliers)} outliers. Removing from dataset...")
    if len(outliers) > 0:
        outlier_indices = outliers['Study'].tolist()
        adult_data_py = adult_data_py.drop(adult_data_py.index[np.array(outlier_indices) - 1])
        
        # Update R dataframe
        with localconverter(ro.default_converter + pandas2ri.converter):
            adult_data_r = ro.conversion.py2rpy(adult_data_py)
        
        # Update in R environment
        ro.r.assign('adult.data.hedges', adult_data_r)
    
    # Save cleaned dataset
    adult_data_py.to_csv(f"{output_dir}/adult_data_cleaned.csv", index=False)
    
    # Run multi-level meta-analysis
    print("Running multi-level meta-analysis...")
    run_meta_analysis_model()
    
    # Create overall plot
    print("Creating overall plot...")
    create_overall_plot(output_dir)
    
    # Create family-specific plots
    print("Creating family-specific plots...")
    create_family_plots(output_dir)
    
    # Extract and save model results
    ro.r(f'''
    # Save model summary to file
    sink("{output_dir}/model_summary.txt")
    print(summary(model.full.adult.hedges.id.nested))
    sink()
    
    # Save model coefficients
    write.csv(coef(summary(model.full.adult.hedges.id.nested)), 
              file="{output_dir}/model_coefficients.csv")
    ''')
    
    print("Meta-analysis complete!")
    return adult_data_py

# Example usage
if __name__ == "__main__":
    # Set the file path to your data
    file_path = "data/meta_2022/adult_raw_data(in).csv"
    
    # Run the complete analysis
    cleaned_data = run_complete_meta_analysis(file_path)
    
    print(f"Cleaned dataset has {len(cleaned_data)} observations")
    print(f"All results have been saved to the output directory")