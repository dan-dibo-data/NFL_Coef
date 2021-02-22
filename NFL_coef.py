import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import Ridge

df = pd.read_csv('fball.csv', skiprows=1, names=['date', 'visitor', 'visitor_points', 'home', 'home_points'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

df['point_difference'] = df['home_points'] - df['visitor_points']
df['home_win'] = np.where(df['point_difference'] > 0, 1, 0)
df['home_loss'] = np.where(df['point_difference'] < 0, 1, 0)

df_visitor = pd.get_dummies(df['visitor'], dtype=np.int64)
df_home = pd.get_dummies(df['home'], dtype=np.int64)

df_model = df_home.sub(df_visitor)
df_model['point_difference'] = df['point_difference']
df_train = df_model
lr = Ridge(alpha=0.001)
X = df_model.drop(['point_difference'], axis=1)
y = df_model['point_difference']
lr.fit(X, y)

df_ratings = pd.DataFrame(data={'team': X.columns, 'rating': lr.coef_})
print(lr.intercept_)
print(df_ratings)