# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Importing DataFrame
housing_df = pd.read_csv("ames_housing.csv")

# Eksik verisi olan sütunların ve eksik veri sayısının tespiti
print(housing_df.isna().sum()[housing_df.isna().sum() > 0].sort_values(ascending=False))

# Eksik verisi olan sütunların veri tipleri
missing_cols = ['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Mas Vnr Type', 'Fireplace Qu',
                'Lot Frontage', 'Garage Yr Blt', 'Garage Cond', 'Garage Qual', 'Garage Finish', 'Garage Type',
                'Bsmt Exposure', 'BsmtFin Type 2', 'Bsmt Qual', 'BsmtFin Type 1', 'Bsmt Cond',
                'Mas Vnr Area', 'Bsmt Half Bath', 'Bsmt Full Bath', 'Electrical', 'Garage Cars',
                'Garage Area', 'Total Bsmt SF', 'Bsmt Unf SF', 'BsmtFin SF 2', 'BsmtFin SF 1']

# Eksik verisi olan sütunları numerical ve categorical olarak ayır
print(housing_df[missing_cols].dtypes)
categorical_missing_cols = ['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Mas Vnr Type', 'Fireplace Qu',
                    'Garage Cond', 'Garage Qual', 'Garage Finish', 'Garage Type', 'Bsmt Exposure',
                    'BsmtFin Type 2', 'Bsmt Qual', 'BsmtFin Type 1', 'Bsmt Cond', 'Electrical']

numerical_missing_cols = ['Lot Frontage', 'Garage Yr Blt', 'Mas Vnr Area', 'Bsmt Half Bath', 'Bsmt Full Bath',
                  'Garage Cars', 'Garage Area', 'Total Bsmt SF', 'Bsmt Unf SF', 'BsmtFin SF 2', 'BsmtFin SF 1']

# Numerical sütunlarda o özellik olmayan evlere 0 de ve modelin özellik olup olmadığını öğrenmesi için flag sütunu ekle
for col in numerical_missing_cols:
    housing_df[col + '_missing'] = housing_df[col].isnull().astype(int)
    housing_df[col] = housing_df[col].fillna(0)

# Eksik verisi olan sütunların ve eksik veri sayısının tekrar kontrolü
print(housing_df.isna().sum()[housing_df.isna().sum() > 0].sort_values(ascending=False))

        # Splitting the data
np.random.seed(42)

X = housing_df.drop("SalePrice", axis=1)
y = housing_df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Encoding
# Categorical sütunları bul
categorical_cols = X_train.select_dtypes(include=['object']).columns
categorical_cols

# Build OneHotEncoder and fit the categorical columns
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(X_train[categorical_cols]);

# Hem train hem de test'teki kategorik sütunları dönüştür
X_train_ohe = pd.DataFrame(ohe.transform(X_train[categorical_cols]),
                                         columns=ohe.get_feature_names_out(categorical_cols),
                                         index=X_train.index)
X_test_ohe = pd.DataFrame(ohe.transform(X_test[categorical_cols]),
                          columns=ohe.get_feature_names_out(categorical_cols),
                          index=X_test.index)

# Numerical sütunları al
numerical_cols = X_train.drop(columns=categorical_cols).columns
X_train_numerical = X_train[numerical_cols]
X_test_numerical = X_test[numerical_cols]

# Numerical ve categorical sütunları birleştir
X_train_final = pd.concat([X_train_numerical, X_train_ohe], axis=1)
X_test_final = pd.concat([X_test_numerical, X_test_ohe], axis=1)

        # Model and Predictions
def evaluate_preds(model, X_train, X_test, y_train, y_test, y_pred):
    from sklearn.metrics import mean_squared_error
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    print(f"Training score: {score_train}")
    print(f"Test score: {score_test}")
    print(f"Mean squared error: {mse}") 
    print(f"Root mean squared error: {rmse}")
    return score_train, score_test, mse, rmse

# Linear Regression Model
linear_regression = LinearRegression()

linear_regression.fit(X_train_final, y_train)

y_pred_lr = linear_regression.predict(X_test_final)
evaluate_preds(linear_regression, X_train_final, X_test_final, y_train, y_test, y_pred_lr)

        # Model Improving
# Ridge Regression
ridge = Ridge()

param_grid = {'alpha': [0.01, 0.1, 1, 10, 50, 100, 200]}

grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train_final, y_train)

print(f"Best params: {grid_search.best_params_}")
best_ridge = grid_search.best_estimator_

y_pred_ridge = best_ridge.predict(X_test_final)
evaluate_preds(best_ridge, X_train_final, X_test_final, y_train, y_test, y_pred_ridge)

# Ridge vs Lasso vs ElasticNet vs Random Forest Regressor vs Gradient Boosting Regressor
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
}

param_grids = {
    'Ridge': {'alpha': [0.1, 1, 10, 100]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1, 1]},
    'ElasticNet': {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.5, 0.9]},
    'RandomForestRegressor': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'GradientBoostingRegressor': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
}

best_models = {}
best_models_score = {}

for name in models:
    print(f"Model: {name}")
    model = models[name]
    param_grid = param_grids[name]

    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=2, scoring='neg_root_mean_squared_error', n_jobs=1)
    grid_search.fit(X_train_final, y_train)

    best_models[name] = grid_search.best_estimator_

    y_pred = best_models[name].predict(X_test_final)

    best_models_score[name] = evaluate_preds(best_models[name], X_train_final, X_test_final, y_train, y_test, y_pred)

best_models_score # Highest training scores and test scores were Random Forest Regressor and Gradient Boosting Regressor

    # Improving Random Forest Regressor and Gradient Boosting Regressor
# GridSearchCV
models = {
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
}

param_grids = {
    'RandomForestRegressor': {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'max_features': ['auto', 'sqrt', 'log2', 0.3, 0.5]},
    'GradientBoostingRegressor': {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.05], 'max_depth': [3, 5], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'max_features': [None, 'sqrt', 'log2'], 'subsample': [0.5, 0.7], 'loss': ['squared_error', 'huber', 'quantile'], 'alpha': [0.3, 0.75]}
}

best_models_grid = {}
best_models_score_grid = {}

for name in models:
    print(f"Model: {name}")
    model = models[name]
    param_grid = param_grids[name]

    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=2, scoring='neg_root_mean_squared_error', n_jobs=1)
    grid_search.fit(X_train_final, y_train)

    best_models_grid[name] = grid_search.best_estimator_

    y_pred = best_models_grid[name].predict(X_test_final)

    best_models_score_grid[name] = evaluate_preds(best_models_grid[name], X_train_final, X_test_final, y_train, y_test, y_pred)

# RandomizedSearchCV
models = {
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
}

param_grids = {
    'RandomForestRegressor': {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'max_features': ['sqrt', 'log2', 0.3, 0.5]},
    'GradientBoostingRegressor': {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'max_features': [None, 'sqrt', 'log2'], 'subsample': [0.5, 0.7], 'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'], 'alpha': [0.1, 0.5, 0.9]}
}

best_models_rs = {}
best_models_score_rs = {}

for name in models:
    print(f"Model: {name}")
    model = models[name]
    param_grid = param_grids[name]

    randomized_search = RandomizedSearchCV(model, param_grid, cv=5, verbose=2, scoring='neg_root_mean_squared_error', n_jobs=1, n_iter=50)
    randomized_search.fit(X_train_final, y_train)

    best_models_rs[name] = randomized_search.best_estimator_

    y_pred = best_models_rs[name].predict(X_test_final)

    best_models_score_rs[name] = evaluate_preds(best_models[name], X_train_final, X_test_final, y_train, y_test, y_pred)

# Comparing Results
print(f"Randomized Search: {best_models_rs}")
print(f"Randomized Search Score: {best_models_score_rs}")
print(f"Grid Search: {best_models_grid}")
print(f"Grid Search Score: {best_models_score_grid}")

    # Creating the best model
from sklearn.ensemble import GradientBoostingRegressor

the_best_model = GradientBoostingRegressor(alpha=0.5, max_depth=5, min_samples_leaf=2, min_samples_split=5, n_estimators=300, random_state=42, subsample=0.7, loss='quantile')

# Fitting the datas to the best model
the_best_model.fit(X_train_final, y_train)

# Best model's predictions and scores
y_pred_best_model = the_best_model.predict(X_test_final)
evaluate_preds(the_best_model, X_train_final, X_test_final, y_train, y_test, y_pred_best_model)

        # Predicting results for different, new values
# Importing new datas
np.random.seed(42)

# Ames Housing dataset all columns
columns_93 = [
    'Order', 'PID', 'MS SubClass', 'MS Zoning', 'Lot Frontage', 'Lot Area',
    'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities', 'Lot Config',
    'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type',
    'House Style', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add',
    'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
    'Mas Vnr Area', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual',
    'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1',
    'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
    'Heating', 'Heating QC', 'Central Air', 'Electrical', '1st Flr SF',
    '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath',
    'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr',
    'Kitchen Qual', 'TotRms AbvGrd', 'Functional', 'Fireplaces', 'Fireplace Qu',
    'Garage Type', 'Garage Yr Blt', 'Garage Finish', 'Garage Cars', 'Garage Area',
    'Garage Qual', 'Garage Cond', 'Paved Drive', 'Wood Deck SF', 'Open Porch SF',
    'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Pool QC',
    'Fence', 'Misc Feature', 'Misc Val', 'Mo Sold', 'Yr Sold', 'Sale Type',
    'Sale Condition', 'SalePrice',
    # Missing value indicator columns (common in this dataset)
    'Lot Frontage_missing', 'Garage Yr Blt_missing', 'Mas Vnr Area_missing',
    'Bsmt Half Bath_missing', 'Bsmt Full Bath_missing', 'Garage Cars_missing',
    'Garage Area_missing', 'Total Bsmt SF_missing', 'Bsmt Unf SF_missing',
    'BsmtFin SF 2_missing', 'BsmtFin SF 1_missing'
]

# Categorical columns list
categorical_cols = [
    'MS Zoning', 'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities',
    'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2',
    'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st',
    'Exterior 2nd', 'Mas Vnr Type', 'Exter Qual', 'Exter Cond', 'Foundation',
    'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2',
    'Heating', 'Heating QC', 'Central Air', 'Electrical', 'Kitchen Qual',
    'Functional', 'Fireplace Qu', 'Garage Type', 'Garage Finish', 'Garage Qual',
    'Garage Cond', 'Paved Drive', 'Pool QC', 'Fence', 'Misc Feature', 'Sale Type',
    'Sale Condition'
]

# Numerical columns list
numerical_cols = [col for col in columns_93 if col not in categorical_cols]

# Values for categorical columns
cat_values = {
    'MS Zoning': ['RL', 'RM', 'FV', 'RH', 'C (all)'],
    'Street': ['Pave', 'Grvl'],
    'Alley': ['Grvl', 'Pave', None],
    'Lot Shape': ['Reg', 'IR1', 'IR2', 'IR3'],
    'Land Contour': ['Lvl', 'Bnk', 'HLS', 'Low'],
    'Utilities': ['AllPub', 'NoSeWa', 'NoSewr'],
    'Lot Config': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
    'Land Slope': ['Gtl', 'Mod', 'Sev'],
    'Neighborhood': ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown'],
    'Condition 1': ['Norm', 'Feedr', 'PosN', 'Artery', 'RRNn', 'RRAe'],
    'Condition 2': ['Norm', 'Feedr', 'PosN', 'Artery', 'RRNn', 'RRAe'],
    'Bldg Type': ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'],
    'House Style': ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer', '2.5Unf'],
    'Roof Style': ['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed'],
    'Roof Matl': ['CompShg', 'Metal', 'WdShake', 'WdShngl', 'Membran'],
    'Exterior 1st': ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'Plywood', 'CemntBd'],
    'Exterior 2nd': ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'Plywood', 'CmentBd'],
    'Mas Vnr Type': ['None', 'BrkFace', 'Stone', 'BrkCmn'],
    'Exter Qual': ['TA', 'Gd', 'Ex', 'Fa'],
    'Exter Cond': ['TA', 'Gd', 'Ex', 'Fa'],
    'Foundation': ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab'],
    'Bsmt Qual': ['TA', 'Gd', 'Ex', 'Fa', None],
    'Bsmt Cond': ['TA', 'Gd', 'Ex', 'Fa', None],
    'Bsmt Exposure': ['No', 'Mn', 'Av', 'Gd', None],
    'BsmtFin Type 1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', None],
    'BsmtFin Type 2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', None],
    'Heating': ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'],
    'Heating QC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Central Air': ['Y', 'N'],
    'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
    'Kitchen Qual': ['TA', 'Gd', 'Ex', 'Fa'],
    'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
    'Fireplace Qu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', None],
    'Garage Type': ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', None],
    'Garage Finish': ['Fin', 'RFn', 'Unf', None],
    'Garage Qual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', None],
    'Garage Cond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', None],
    'Paved Drive': ['Y', 'P', 'N'],
    'Pool QC': ['Ex', 'Gd', 'TA', 'Fa', None],
    'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None],
    'Misc Feature': ['Elev', 'Gar2', 'Shed', 'TenC', None],
    'Sale Type': ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'ConLw', 'CWD', 'VWD', 'Oth'],
    'Sale Condition': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'],
}

# Ranges for numerical columns
num_ranges = {
    'Order': (1, 1460),
    'PID': (1000000, 1001460),
    'MS SubClass': [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
    'Lot Frontage': (20, 120),
    'Lot Area': (2000, 20000),
    'Overall Qual': (1, 10),
    'Overall Cond': (1, 10),
    'Year Built': (1872, 2010),
    'Year Remod/Add': (1950, 2010),
    'Mas Vnr Area': (0, 500),
    'BsmtFin SF 1': (0, 1500),
    'BsmtFin SF 2': (0, 1000),
    'Bsmt Unf SF': (0, 1000),
    'Total Bsmt SF': (300, 3000),
    '1st Flr SF': (300, 3000),
    '2nd Flr SF': (0, 1500),
    'Low Qual Fin SF': (0, 500),
    'Gr Liv Area': (300, 4000),
    'Bsmt Full Bath': (0, 3),
    'Bsmt Half Bath': (0, 2),
    'Full Bath': (0, 4),
    'Half Bath': (0, 2),
    'Bedroom AbvGr': (0, 8),
    'Kitchen AbvGr': (0, 3),
    'TotRms AbvGrd': (2, 15),
    'Fireplaces': (0, 4),
    'Garage Yr Blt': (1900, 2010),
    'Garage Cars': (0, 4),
    'Garage Area': (0, 1500),
    'Wood Deck SF': (0, 1000),
    'Open Porch SF': (0, 500),
    'Enclosed Porch': (0, 500),
    '3Ssn Porch': (0, 300),
    'Screen Porch': (0, 300),
    'Pool Area': (0, 800),
    'Misc Val': (0, 10000),
    'Mo Sold': (1, 12),
    'Yr Sold': (2006, 2010),
    'SalePrice': (50000, 500000),
    # Missing indicator for numerical columns
    'Lot Frontage_missing': (0, 1),
    'Garage Yr Blt_missing': (0, 1),
    'Mas Vnr Area_missing': (0, 1),
    'Bsmt Half Bath_missing': (0, 1),
    'Bsmt Full Bath_missing': (0, 1),
    'Garage Cars_missing': (0, 1),
    'Garage Area_missing': (0, 1),
    'Total Bsmt SF_missing': (0, 1),
    'Bsmt Unf SF_missing': (0, 1),
    'BsmtFin SF 2_missing': (0, 1),
    'BsmtFin SF 1_missing': (0, 1),
}

# MS SubClass için olası değerler
ms_subclass_values = num_ranges['MS SubClass']

# Creating data dictionary
data = {col: [] for col in columns_93}

for i in range(3):
    for col in columns_93:
        if col in categorical_cols:
            vals = cat_values.get(col, [None])
            data[col].append(np.random.choice(vals))
        else:
            if col == 'MS SubClass':
                data[col].append(np.random.choice(ms_subclass_values))
            elif col in num_ranges:
                low, high = num_ranges[col]
                # 0 or 1 for missing indicator columns randomly
                if col.endswith('_missing'):
                    val = np.random.randint(0, 2)
                elif col in ['Order', 'PID', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add',
                             'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr',
                             'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 'Garage Cars',
                             'Mo Sold', 'Yr Sold']:
                    val = np.random.randint(low, high + 1)
                else:
                    val = np.round(np.random.uniform(low, high), 2)
                data[col].append(val)
            else:
                data[col].append(None)

new_houses = pd.DataFrame(data)

# Checking empty values and filling
new_houses.isna().sum()[new_houses.isna().sum() > 0]
missing_cols_new = ['Alley', 'Bsmt Cond', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Fireplace Qu', 'Garage Qual', 'Garage Cond', 'Fence', 'Misc Feature']
new_houses_splitted = new_houses.drop("SalePrice", axis=1)

categorical_missing_cols_new = ['Alley', 'Bsmt Cond', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Fireplace Qu', 'Garage Qual', 'Garage Cond', 'Fence', 'Misc Feature']

# Categorical sütunlardaki eksik verileri yani o özellik bulunmayan evlere 'None' de
for col in categorical_missing_cols_new:
    new_houses_splitted[col] = new_houses_splitted[col].fillna('None')

# Eksik verisi olan sütunların ve eksik veri sayısının tekrar kontrolü
print(new_houses_splitted.isna().sum()[new_houses_splitted.isna().sum() > 0].sort_values(ascending=False))
print(new_houses_splitted.dtypes.value_counts())

    # Encoding
from sklearn.preprocessing import OneHotEncoder
# Hem train hem de test'teki kategorik sütunları dönüştür
new_houses_ohe = pd.DataFrame(ohe.transform(new_houses_splitted[categorical_cols]),
                                         columns=ohe.get_feature_names_out(categorical_cols),
                                         index=new_houses_splitted.index)

# Numerical sütunları al
numerical_cols = new_houses_splitted.drop(columns=categorical_cols).columns
new_houses_numerical = new_houses_splitted[numerical_cols]

# Numerical ve categorical sütunları birleştir
new_houses_final = pd.concat([new_houses_numerical, new_houses_ohe], axis=1)

# New predictions
new_houses_predictions = the_best_model.predict(new_houses_final)
new_houses_predictions

# Comparing a real price in the test set with that house's prediction price
yeni_ev = X_test_final.iloc[[3]]
yeni_ev_prediction = the_best_model.predict(yeni_ev)
yeni_ev_prediction, y_test.iloc[3]