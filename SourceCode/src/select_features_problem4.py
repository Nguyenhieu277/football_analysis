import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'D:\work\football_analysis\SourceCode\data\players_played_more_than_900m.csv')

feature_names = df.drop(['player', 'team', 'nation', 'Value'], axis=1).columns.tolist()  
X = df[feature_names]

y = df['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  

model = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)

model.fit(X_train, y_train)
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Most Important Features')
plt.savefig('RandomForestRegressor.png')
plt.show()

numeric_df = df.select_dtypes(include=['number'])

correlation = numeric_df.corr()
correlation_with_target = abs(correlation['Value']).sort_values(ascending=False)

plt.figure(figsize=(12, 10))
top_corr_features = correlation.index[abs(correlation["Value"]) > 0.3]
plt.title('Correlation Heatmap for Features with |Correlation| > 0.3')
sns_plot = sns.heatmap(numeric_df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

n_features_to_select = 15
rfe = RFE(estimator=RandomForestRegressor(n_estimators=1000, random_state=42), 
          n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)
selected_features_rfe = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]

k_best = 15
selector = SelectKBest(f_regression, k=k_best)
selector.fit(X_train, y_train)
selected_features_kbest = [feature_names[i] for i in range(len(feature_names)) 
                           if selector.get_support()[i]]

kbest_scores = pd.DataFrame({
    'Feature': feature_names,
    'Score': selector.scores_
})
kbest_scores = kbest_scores.sort_values('Score', ascending=False)

top_rf_features = feature_importance_df['Feature'].head(15).tolist()

common_features = set(top_rf_features) & set(selected_features_rfe) & set(selected_features_kbest)

all_selected_features = list(set(top_rf_features + selected_features_rfe + selected_features_kbest))
comparison_df = pd.DataFrame(index=all_selected_features, columns=['Random Forest', 'RFE', 'SelectKBest'])

for feature in all_selected_features:
    comparison_df.loc[feature, 'Random Forest'] = feature in top_rf_features
    comparison_df.loc[feature, 'RFE'] = feature in selected_features_rfe
    comparison_df.loc[feature, 'SelectKBest'] = feature in selected_features_kbest

comparison_df['Count'] = comparison_df.sum(axis=1)
comparison_df = comparison_df.sort_values('Count', ascending=False)

plt.figure(figsize=(12, len(all_selected_features) * 0.5))
sns.heatmap(comparison_df.drop('Count', axis=1).astype(int), cmap='YlGnBu', cbar=False, annot=True)
plt.title('Features Selected by Different Methods')
plt.tight_layout()
plt.savefig('feature_selection_comparison.png')

if common_features:
    plt.figure(figsize=(10, 6))
    common_importance = feature_importance_df[feature_importance_df['Feature'].isin(common_features)]
    sns.barplot(x='Importance', y='Feature', data=common_importance)
    plt.title('Common Features Selected by All Methods')
    plt.tight_layout()
    plt.savefig('common_features.png')

plt.show()