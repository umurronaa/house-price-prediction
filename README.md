# House Price Prediction

This project aims to predict house prices using the **Ames Housing Dataset** with various regression models and hyperperameter optimization techniques.

## Project Structure

house-price-prediction/
- ├── ames_housing.csv # Dataset
- ├── project.ipynb # Jupyter Notebook version of the project
- ├── project.py script version
- ├── requirements.txt # Python dependencies
- └── README.md # Project description

## Models used

The following regression models are trained and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor

Both `GridSearchCV`and `RandomizedSearchCV` are used for hyperparameter tuning

## Dataset
- Dataset: [Ames Housing Dataset - Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- Target variable: `SalePrice`

## Results

Many hyperparameters were tried. The best performing model was `GradientBoostingRegressor` with hyperparameters
```python
alpha = 0.5
max_depth = 5
min_samples_leaf = 2
min_samples_split = 5
n_estimators = 300
random_state = 42
subsample = 0.7
loss = 'quantile'
```

# Features

- Missing value handling
- Categorical variable encoding via OneHotEncoder
- Hyperparameter tuning via GridSearchCV & RandomizedSearchCV
- Model comparison and selection
- Prediction on new, synthetic house data

# Author
Developed by Umur Rona - umurmrona@gmail.com