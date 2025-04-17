import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import randint, uniform 
import time
import xgboost as xgb
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'data/players_played_more_than_900m.csv')

X = df[['age', 'xG', 'Att Pen', 'SCA', 'xG/90', 'PrgP/passing', 'PPA', 'SoT/90', 'GCA', 'Att 3rd', 'Medium_Cmp%', 'ProDist', 'Cmp%']]
y = df['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tuning_config = {
    "LinearRegression": {
        "model": LinearRegression(n_jobs=-1),
        "param_dist": {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
        },
        "param_grid": lambda bp: {
            'fit_intercept': [bp.get('fit_intercept', True)],
            'copy_X': [bp.get('copy_X', True)],
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "param_dist": {
            'n_estimators': randint(low=100, high=800),
            'max_depth': randint(low=10, high=50),
            'min_samples_split': randint(low=2, high=10),
            'min_samples_leaf': randint(low=1, high=5),
        },
        "param_grid": lambda bp: { 
            'n_estimators': sorted(list(set([max(10, bp.get('n_estimators', 300) - 50), bp.get('n_estimators', 300), bp.get('n_estimators', 300) + 50]))),
            'max_depth': sorted(list(set([bp.get('max_depth', 10) - 5 if bp.get('max_depth', 10) is not None else None, bp.get('max_depth', 10), bp.get('max_depth', 10) + 5 if bp.get('max_depth', 10) is not None else None, None])), key=lambda x: (x is None, x)),
            'min_samples_split': sorted(list(set([max(2, bp.get('min_samples_split', 5) - 1), bp.get('min_samples_split', 5), bp.get('min_samples_split', 5) + 1]))),
            'min_samples_leaf': sorted(list(set([max(1, bp.get('min_samples_leaf', 3) - 1), bp.get('min_samples_leaf', 3), bp.get('min_samples_leaf', 3) + 1]))),
        }
    },
    "XGBoost" : {
        "model": xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
        "param_dist": {
            'n_estimators': randint(low=100, high=800),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(low=3, high=12),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': [0, 0.1, 0.5, 1],
            'reg_alpha': [0, 0.001, 0.01, 0.1],
            'reg_lambda': [0.1, 0.5, 1, 5]
        },
        "param_grid": lambda bp: {
             'n_estimators': sorted(list(set([max(10, bp.get('n_estimators', 300) - 50), bp.get('n_estimators', 300), bp.get('n_estimators', 300) + 50]))),
             'learning_rate': sorted(list(set([max(0.01, bp.get('learning_rate', 0.1) / 2), bp.get('learning_rate', 0.1), min(0.3, bp.get('learning_rate', 0.1) * 1.5)]))),
             'max_depth': sorted(list(set([max(1, bp.get('max_depth', 5) - 1), bp.get('max_depth', 5), bp.get('max_depth', 5) + 1]))),
             'subsample': [bp.get('subsample', 0.8)],
             'colsample_bytree': [bp.get('colsample_bytree', 0.8)],
             'gamma': [bp.get('gamma', 0)],
             'reg_alpha': [bp.get('reg_alpha', 0)],
             'reg_lambda': [bp.get('reg_lambda', 1)]
        }
    }
}
n_folds = 5
n_iter_random = 50
results = {}
best_estimators = {}
tuning_times = {}

for model_name, config in tuning_config.items():
    total_start_time = time.time()

    random_search = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["param_dist"],
        n_iter=n_iter_random,
        cv=n_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    best_params_random = random_search.best_params_
    best_score_random = -random_search.best_score_

    param_grid = config["param_grid"](best_params_random)
    for key, values in param_grid.items():
        valid_values = []
        seen = set()
        if key == 'n_estimators': valid_values = [max(10, v) for v in values]
        elif key == 'max_depth': valid_values = [v if v is None else max(1, v) for v in values]
        elif key == 'min_samples_split': valid_values = [max(2, v) for v in values]
        elif key == 'min_samples_leaf': valid_values = [max(1, v) for v in values]
        elif key == 'learning_rate': valid_values = [max(0.001, v) for v in values]
        else: valid_values = values

        unique_valid_values = []
        for v in sorted(valid_values, key=lambda x: (x is None, x)):
             v_hashable = tuple(v) if isinstance(v, list) else v
             if v_hashable not in seen:
                 unique_valid_values.append(v)
                 seen.add(v_hashable)
        param_grid[key] = unique_valid_values

    grid_search = GridSearchCV(
        estimator=config["model"],
        param_grid=param_grid,
        cv=n_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_params_grid = grid_search.best_params_
    best_score_grid = -grid_search.best_score_
    best_estimator = grid_search.best_estimator_

    results[model_name] = {'Best MAE (Tuned CV)': best_score_grid, 'Best Params': best_params_grid}
    best_estimators[model_name] = best_estimator
    total_end_time = time.time()
    tuning_times[model_name] = total_end_time - total_start_time

results_df = pd.DataFrame(results).T
results_df['Total Tuning Time (s)'] = pd.Series(tuning_times)
results_df = results_df.sort_values(by='Best MAE (Tuned CV)')

overall_best_model_name = results_df.index[0]
overall_best_estimator = best_estimators[overall_best_model_name]

y_pred_test_final = overall_best_estimator.predict(X_test)
final_mae = mean_absolute_error(y_test, y_pred_test_final)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_final))

print(f"Final Results - Best Model: {overall_best_model_name}, MAE: {final_mae:.2f}, RMSE: {final_rmse:.2f}")
