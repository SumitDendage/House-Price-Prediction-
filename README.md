# House Price Prediction (India)

Regression models that estimate house prices from 22 property features — size, rooms, condition, grade, location, schools nearby and distance to the airport — using a dataset of 14,620 Indian housing sales.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines%20%7C%20GridSearchCV-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Workflow

1. **Exploration** – distributions of every feature, a correlation heat-map and a latitude/longitude price map.
2. **Cleaning** – drop missing values and remove price outliers with the IQR rule (14,620 → 13,859 rows).
3. **Modelling** – each model is a scikit-learn `Pipeline` of `StandardScaler` + regressor, trained on an 80/20 split.
4. **Tuning** – `GridSearchCV` (3-fold) over *k* = 1…10 for K-Nearest Neighbours.
5. **Evaluation** – R², MAE and RMSE on the test set, plus predicted-vs-actual plots with a fitted line.

## Results (test set)

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.94 | 35,612 | 48,896 |
| K-Nearest Neighbours | 0.87 | 51,399 | 73,307 |
| K-Nearest Neighbours (tuned *k*) | 0.88 | 50,243 | 72,401 |
| Random Forest | ≈1.00 | 78 | 202 |

Linear Regression gives the most trustworthy result. The Random Forest's near-perfect score is too good to be true on real sales data and should be checked for leakage (for example by dropping the `id` column and scoring with cross-validation) before relying on it.

## Dataset

`House Price India.csv` — 14,620 sales with 23 columns, including bedrooms, bathrooms, living and lot area, floors, waterfront, views, condition, grade, built and renovation year, postal code, latitude/longitude, number of schools nearby, distance from the airport and **Price** (target).

## Run it

```bash
git clone https://github.com/SumitDendage/House-Price-Prediction-.git
cd House-Price-Prediction-
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Updated_House_Price_Prediction.ipynb
```

`root_mean_squared_error` needs scikit-learn 1.4 or newer.

## Author

**Sumit Dendage** · [GitHub](https://github.com/SumitDendage) · [LinkedIn](https://www.linkedin.com/in/sumit-dendage-636a50201) · [Medium](https://dendagesumitbalkrishna123.medium.com)
