# 🏠 DataDiscovery: Housing Data Analysis Toolkit

Comprehensive exploration and analysis of housing datasets with Python.

## 📊 Description

Exploring and understanding a housing dataset. It provides a comprehensive suite of analysis methods that help uncover patterns, correlations, and insights within real estate data.

## ✨ Features

- **Comprehensive Data Overview**: Quickly understand dataset dimensions, column types, and basic statistics
- **Missing Value Analysis**: Identify and quantify null values and zeros across all features
- **Distribution Analysis**: Visualize and analyze the distribution of key variables like sale prices
- **Correlation Analysis**: Generate correlation matrices and heatmaps to identify relationships between features
- **Outlier Detection**: Identify potential outliers using standardized scaling
- **Bivariate Analysis**: Explore relationships between pairs of variables
- **Categorical Analysis**: Analyze categorical variables through grouping and counting

## 🛠️ Setup Guide

### Prerequisites

- Python 3.x
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/DataDiscovery.git
   cd DataDiscovery
   ```

2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

## 📋 Usage

The DataDiscovery class automatically runs a comprehensive analysis when instantiated:

```python
from main import DataDiscovery

# Initialize with default dataset (train.csv)
analysis = DataDiscovery()

# All analysis methods are run automatically during initialization
# Results are printed to console and visualizations are displayed
```
## 📊 Example Analyses

### Distribution Analysis
Analyzes the distribution of the target variable (SalePrice) including:
- Basic statistics (min, max, mean, median)
- Skewness and kurtosis
- Distribution visualization
- Box plots by quality rating

### Correlation Analysis
Generates correlation matrices to identify the most important features related to sale prices:
- Heatmap of all feature correlations
- Focused analysis on top correlated features

### Outlier Detection
Identifies potential outliers in the dataset using standardized scaling, helping to clean data for more accurate modeling.

## 🔍 Resources

For more information on the housing dataset used in this project, visit:
- [Kaggle Housing Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## 📝 License

This project is licensed under the terms included in the LICENSE file.
