import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Tạo thư mục lưu biểu đồ và mô hình nếu chưa tồn tại
output_folder = 'resources'
model_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Tải dữ liệu
datasets = {
    'FPT': pd.read_csv('data/merged_FPT_data.csv'),
    'HPG': pd.read_csv('data/merged_HPG_data.csv'),
    'VCB': pd.read_csv('data/merged_VCB_data.csv'),
    'VIC': pd.read_csv('data/merged_VIC_data.csv'),
    'VNM': pd.read_csv('data/merged_VNM_data.csv')
}

# Cấu hình tinh chỉnh mô hình
tuning_params = {
    'Ridge Regression': {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]},
    'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
    'Random Forest Regressor': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]},
    'XGBoost Regressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1]}
}

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'XGBoost Regressor': XGBRegressor(objective='reg:squarederror', random_state=42)
}


# Hàm tinh chỉnh và đánh giá mô hình
def tune_and_evaluate(X, y, model, params):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    if params:
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=kf, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X, y)
    cv_rmse = (-cross_val_score(best_model, X, y, cv=kf, scoring='neg_root_mean_squared_error').mean()) ** 0.5
    cv_r2 = cross_val_score(best_model, X, y, cv=kf, scoring='r2').mean()
    return best_model, cv_rmse, cv_r2


# Ánh xạ mô hình vào bộ datasets
results = []
for dataset_name, data in datasets.items():
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)
    target_column = [col for col in data.columns if 'close' in col.lower()][0]
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Chuẩn hóa features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for model_name, model in models.items():
        params = tuning_params.get(model_name, {})
        best_model, cv_rmse, cv_r2 = tune_and_evaluate(X_scaled, y, model, params)
        results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'CV_RMSE': cv_rmse,
            'CV_R^2': cv_r2
        })

        # Save the tuned model
        joblib.dump(best_model, os.path.join(model_folder, f'{model_name}_{dataset_name}_tuned.joblib'))

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Lưu kết quả vào CSV
results_df.to_csv('model_comparison_results.csv', index=False)
