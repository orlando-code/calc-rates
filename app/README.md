# Metafor Meta-Analysis Streamlit App

Interactive web application for meta-analysis using the existing `calcification/analysis` codebase with robust R integration.

## Features

- 🔬 **Meta-Analysis**: Uses your existing `MetaforModel` class with robust context management
- 📊 **Interactive Exploration**: Filter by treatment, calcification units, and moderator variables  
- 📈 **Visualization**: Effect size distributions and moderator relationships
- 🔧 **Flexible Modeling**: Choose effect types, random effects structures, and moderators
- 💾 **Data Export**: Download filtered datasets for further analysis

## Quick Start

### Prerequisites
- Ensure you're in the `calcer` conda environment
- R and the `metafor` package should be installed

### Installation
```bash
# Navigate to the app directory
cd app/

# Install Python dependencies (if not already installed)
pip install -r requirements.txt

# Launch the app
python run_app.py
```

The app will open in your browser at `http://localhost:8501`.

## How It Works

### Robust R Integration
The app uses a `RobustMetaforModel` wrapper around your existing `MetaforModel` class that:

1. **Context Management**: Ensures all R operations are wrapped in proper conversion contexts
2. **Thread Safety**: Initializes R in the current thread to avoid Streamlit threading issues
3. **Error Handling**: Provides detailed error messages and graceful fallbacks

### Key Components

- **`RobustMetaforModel`**: Context-safe wrapper around `calcification.analysis.meta_regression.MetaforModel`
- **Data Loading**: Uses `data/clean/analysis_ready_data.csv` 
- **Interactive Filtering**: Treatment, calcification units, moderator variables
- **Model Configuration**: Effect types, random effects, formula building

### Supported Effect Types
- `st_relative_calcification` (with `st_relative_calcification_var`)
- `hedges_g` (with `hedges_g_var`)

### Supported Moderators
- `delta_ph`: pH change
- `delta_t`: Temperature change  
- `temp`: Temperature
- `phtot`: Total pH

## Usage

1. **Select Effect Type**: Choose between standardized calcification rates or Hedges' g
2. **Filter Data**: Select treatments and calcification units of interest
3. **Choose Moderator**: Optional moderator variable for meta-regression
4. **Configure Model**: Set random effects structure and formula
5. **Fit Model**: Run the meta-analysis with robust R integration
6. **Explore Results**: View model summary, coefficients, and diagnostics

## Troubleshooting

### R Context Errors
If you encounter `NotImplementedError` related to R conversion rules:
- The app automatically handles this with robust context management
- Check that you're in the correct conda environment
- Ensure R and metafor are properly installed

### Import Errors
```bash
# Make sure you're in the calcer environment
conda activate calcer

# Install missing packages
pip install streamlit plotly

# For R packages (run in R console)
install.packages("metafor")
```

### Data Loading Issues
- Ensure `data/clean/analysis_ready_data.csv` exists in the project root
- Check that required columns (`st_relative_calcification`, `hedges_g`, etc.) are present

## Development

### Architecture
```
app/
├── metafor_streamlit_app.py    # Main Streamlit application
├── run_app.py                  # Launch script  
├── requirements.txt            # Python dependencies
└── README.md                   # This file

calcification/analysis/         # Your existing analysis code
├── meta_regression.py          # MetaforModel class (used by app)
├── analysis_utils.py           # Utility functions
└── model_specs.yaml           # Model specifications
```

### Extending the App
- **New Effect Types**: Add to `available_effect_types` list in main()
- **New Moderators**: Add to `potential_moderators` list  
- **Custom Plots**: Extend `create_effect_plot()` function
- **Model Diagnostics**: Add to the results display section

The app is designed to leverage your existing, well-tested analysis code while providing a user-friendly interface with robust R integration.