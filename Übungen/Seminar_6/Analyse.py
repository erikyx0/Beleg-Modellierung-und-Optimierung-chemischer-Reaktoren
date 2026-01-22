import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomTreesEmbedding, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

data = pd.read_csv("Daten/Solid_Holdup.csv")
print(f"Datashape: {data.shape}")
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
data = data.drop_duplicates(keep="first")

#Prepare Features
feature_cols = ["Us", "Ul", "Ut", "D", "x", "x/L", "r/R"]
X = data[feature_cols] #input features
y = data["Solids Holdup"] #output features

summary = pd.DataFrame({
    "Feature": data.columns,
    "Min": data.min().values,
    "Max": data.max().values,
    #"Mean": data.mean().values,
    "Range": (data.max()-data.min()).values
})
print(summary)

def data_scaling(data,min_vals,max_vals):
    """
    Scale Data to range [0, 1] using the formula:
    scales_value = (value - min) / (max - min)
    :param data: Dataframe or array
    :param min_vals: minimum values for each feature
    :param max_vals: maximum values for each feature
    :return: Scaled Dataframe or array
    """
    ranges = max_vals - min_vals
    return (data - min_vals) / ranges

min_vals, max_vals = X.min(), X.max()

X = data_scaling(X,min_vals,max_vals)
X = pd.DataFrame(X, columns = feature_cols)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "K-Neighbors": KNeighborsRegressor(),
    "SVR": SVR()
}

results = []
predictions = {}
for name, model in models.items():
    #Modell trainieren
    model.fit(X_train, y_train)

    #Vorhersagen
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    #Auswertung
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Ergebnisse speichern
    results.append({
        "Model": name,
        "MSE": round(mse,4),
        "RMSE": round(rmse,4),
        "R2": round(r2,4)
    })

results_df = pd.DataFrame(results)
print("\n"*5)
print("="*50)
print(results_df)

#grafische Darstellung
fix,axes = plt.subplots(2,3, figsize = (18,12))

axes = axes.ravel() #?

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
for idx, (name, y_pred) in enumerate(predictions.items()):
    axes[idx].scatter(y_test, y_pred, alpha = 0.6, edgecolor = 'k', s=50, color=colors[idx])
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes[idx].set_title(f"{name}")
    axes[idx].grid()

plt.show(dpi=700)