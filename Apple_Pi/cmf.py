import numpy as np
import pandas as pd

import csv

from cmfrec import CMF

ratings = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\Apple_Pi\\df_rearranged.csv")


model = CMF(method="als", k=26, lambda_=1e+1)
model.fit(ratings)

print(ratings.head())

exclude = ratings.ItemId.loc[ratings.UserId == 777]
print(exclude)

recommended_app = model.topN(user=777, n=5)

print(recommended_app)

app_names = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\Apple_Pi\\df_app_names.csv")

for i in recommended_app:
    print(app_names.AppName.loc[app_names.AppID == i])



