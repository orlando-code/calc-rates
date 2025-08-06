# 🦠 Metafor Analysis Dashboard

An interactive web application for exploring meta-regression results from calcification studies. Built with Streamlit and integrated with your existing metafor analysis pipeline.

## Features

### 🎯 Interactive Analysis
- **Treatment Selection**: Filter by specific treatment types
- **Formula Building**: Build custom regression formulae with predictors and interactions
- **Subset Filtering**: Filter by calcification units, core groupings, and other variables
- **Real-time Model Fitting**: Fit metafor models on-demand with selected parameters
- **R Context Management**: Proper handling of rpy2 threading issues in Streamlit

### 📊 Visualization Options
- **Single Model Plots**: Detailed meta-regression plots with confidence intervals
- **Model Comparison**: Side-by-side comparison of different models
- **Interactive Plotly Charts**: Hover information and zoom capabilities
- **Publication-ready Matplotlib Plots**: High-quality static plots

### ⚖️ Comparison Modes
- **Treatment Comparison**: Compare effects across different treatments
- **Formula Comparison**: Compare different model specifications
- **Unit Comparison**: Compare results across calcification units

### 📈 Data Exploration
- **Summary Statistics**: Key metrics and sample sizes
- **Data Preview**: Browse filtered datasets
- **Distribution Plots**: Explore effect size distributions
- **Export Options**: Download filtered data and results

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements_app.txt
```

### 2. Launch the App
```bash
python run_app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Prepare Your Data
Ensure your processed calcification data is available as `processed_calcification_data.csv` in the project root, or update the data path in the app configuration.

## Usage Guide

### Setting Up Analysis

1. **Select Effect Type**: Choose between standardized relative or absolute calcification
2. **Filter Treatments**: Select one or more treatment types to include
3. **Choose Predictors**: Select variables like `delta_ph`, `delta_t` for your model
4. **Build Formula**: The app automatically generates the regression formula
5. **Set Moderator**: Choose which variable to plot on the x-axis

### Running Single Model Analysis

1. Go to the "🎯 Single Model" tab
2. Configure your selections in the sidebar
3. Click "Run Analysis" to fit the model
4. View the regression plot and model summary

### Comparing Models

1. Go to the "⚖️ Compare Models" tab
2. Choose comparison type (treatments, formulae, or units)
3. Click the appropriate comparison button
4. View side-by-side plots and results

### Exploring Data

1. Go to the "📈 Data Explorer" tab
2. View summary statistics for your filtered dataset
3. Browse the data preview table
4. Examine effect size distributions

## Configuration

### App Configuration (`app_config.yaml`)
- Data file paths and caching settings
- Model fitting parameters
- Plot styling options
- Export settings

### Model Specifications
The app uses your existing `model_specs.yaml` for:
- Available effect types
- Predefined model formulae
- Intercept and grouping options

## File Structure

```
├── metafor_analysis_app.py      # Main Streamlit application
├── streamlit_helpers.py         # Helper functions and utilities
├── run_app.py                   # App launcher script
├── requirements_app.txt         # Python dependencies
├── app_config.yaml             # App configuration
└── README_metafor_app.md       # This documentation
```

## Integration with Existing Code

The app seamlessly integrates with your existing codebase:

- **MetaforModel**: Uses your `calcification.analysis.meta_regression.MetaforModel` class
- **Plotting**: Leverages `calcification.plotting.analysis.MetaRegressionPlotter`
- **Data Processing**: Uses `calcification.analysis.analysis_utils` for preprocessing
- **Configuration**: Reads from your existing `model_specs.yaml`

## Advanced Features

### Custom Formulae
Build complex formulae with:
- Multiple predictors
- Interaction terms
- Intercept control
- Automatic syntax validation

### Model Comparison
Compare up to 4 models simultaneously across:
- Different treatment subsets
- Alternative model specifications
- Various data filtering criteria

### Export Options
- Download filtered datasets as CSV
- Save model results and summaries
- Export high-resolution plots (planned)

## Troubleshooting

### Common Issues

**"Data file not found"**
- Ensure `processed_calcification_data.csv` exists in the project root
- Update the data path in `app_config.yaml`

**"Model fitting failed"**
- Check that selected predictors exist in your data
- Ensure sufficient data points after filtering
- Verify no constant predictors (zero variance)

**"Import errors"**
- Ensure your calcification package is in Python path
- Install missing dependencies from `requirements_app.txt`

### Performance Tips

- Use data filtering to reduce model fitting time
- Limit model comparisons to 3-4 models for better performance
- Clear browser cache if experiencing display issues

## Extending the App

### Adding New Features

1. **Custom Visualizations**: Add new plot types in `streamlit_helpers.py`
2. **Additional Filters**: Extend filtering options in the sidebar
3. **Export Formats**: Add new export formats (PDF, PNG, etc.)
4. **Model Diagnostics**: Integrate diagnostic plots and statistics

### Integration Points

- `load_data()`: Modify to load from different sources
- `create_model_comparison_plot()`: Customize comparison visualizations
- `validate_data_for_model()`: Add custom validation rules

## Dependencies

### Core Requirements
- `streamlit`: Web application framework
- `plotly`: Interactive plotting
- `matplotlib`: Static plotting
- `pandas`: Data manipulation
- `numpy`: Numerical computing

### Your Existing Stack
- `rpy2`: R integration for metafor
- Your `calcification` package modules

## Support

For issues related to:
- **App functionality**: Check the troubleshooting section
- **Model fitting**: Refer to your existing metafor documentation
- **Data processing**: Use your existing calcification package documentation

## Future Enhancements

Planned features:
- 📄 PDF report generation
- 🔄 Batch model processing
- 💾 Session state persistence
- 🌐 Multi-user deployment options
- 📊 Advanced diagnostic plots