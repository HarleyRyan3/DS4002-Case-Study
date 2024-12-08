#import statements
import statsmodels.api as sm
import pandas as pd
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

#probit model code
df_cfb = pd.read_csv('/Users/pallavi/Desktop/CFB_Analysis_Data.csv')

df_cfb = df_cfb.replace([np.inf, -np.inf], np.nan).dropna()
X = df_cfb[['homeElo','awayElo','homeFlairs', 'awayFlairs', 'homeP5', 'notreDame']]
X = sm.add_constant(X)

X = np.asarray(X)
y = df_cfb['flagship'].astype(float)

#Fitting probit model
probit = sm.Probit(y, X)
pr = probit.fit()
print(pr.summary())
#
#
#
#
#
#
#probit model code 2: home and away Elo
df_cfb = pd.read_csv('/Users/pallavi/Desktop/CFB_Analysis_Data.csv')

df_cfb = df_cfb.replace([np.inf, -np.inf], np.nan).dropna()
A = df_cfb[['homeElo','awayElo']]
A = sm.add_constant(A)

A = np.asarray(A)
b = df_cfb['flagship'].astype(float)

#Fitting probit model
probit = sm.Probit(b, A)
pr = probit.fit()
print(pr.summary())
#
#
#
#
#
#
#probit model code 2: home and away flair
df_cfb = pd.read_csv('/Users/pallavi/Desktop/CFB_Analysis_Data.csv')

df_cfb = df_cfb.replace([np.inf, -np.inf], np.nan).dropna()
C = df_cfb[['homeFlairs','awayFlairs']]
C = sm.add_constant(C)

C = np.asarray(C)
d = df_cfb['flagship'].astype(float)

#Fitting probit model
probit = sm.Probit(d, C)
pr = probit.fit()
print(pr.summary())