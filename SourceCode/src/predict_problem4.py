import pandas as pd
import joblib 


df = pd.read_csv(r'D:\work\football_analysis\SourceCode\data\players_played_more_than_900m.csv')

X = df[['age', 'xG', 'Att Pen', 'SCA', 'xG/90', 'PrgP/passing', 'PPA', 'SoT/90', 'GCA', 'Att 3rd', 'Medium_Cmp%', 'TklW', 'ProDist', 'Cmp%']]
y = df['Value']


model_filename = r'D:\work\football_analysis\SourceCode\trained_model\best_randomforest_model.pkl' 
loaded_model = joblib.load(model_filename)


predictions = loaded_model.predict(X)


df['Predicted_Value'] = predictions
df['loss'] = df['Value'] - df['Predicted_Value'] 
print("Predictions made and added to DataFrame:")
print(df[['player', 'Value', 'Predicted_Value', 'loss']].head()) 

df[['player', 'Value', 'Predicted_Value', 'loss']].to_csv(r'D:\work\football_analysis\SourceCode\data\predictions_output.csv', index=False) 
