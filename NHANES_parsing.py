# %% [markdown]
"""
# Parsing NHANES data
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_path = Path("Data/NHANES")

# %% [markdown]
"""
## Loading the data
"""

# %%
df_nhanes = pd.read_csv(data_path / "MainTable.csv")
df_nhanes

# %%
df_vars = pd.read_csv(data_path / "VarDescription.csv")
df_vars

# %% [markdown]
"""
## Data Wrangling
### Drop variables from dietary and supplement use:
"""

# %%
diet_supp_vars = (df_vars['tab_name'].str.startswith("food_consumption") |
                  df_vars['tab_name'].str.startswith("supplement"))
diet_supp_vars = df_vars.loc[diet_supp_vars, 'var']
df_vars = df_vars.query("not var in @diet_supp_vars")
df_nhanes = df_nhanes.loc[:, ~df_nhanes.columns.isin(diet_supp_vars)]
print(df_vars.shape, df_nhanes.shape)

# %% [markdown]
"""
### Filter first two survey years based on eligibility of mortality data (2006)
Survey year (1=1999-2000, 2=2001-2002, 3=2003-2004, 4=2005-2006)
`SDDSRVYR in [1,2]` 

We take only variables meassured in BOTH surveys
"""

# %%
df_vars = df_vars.query("series in ['2001-2002', '1999-2000']")
df_nhanes = df_nhanes.query("SDDSRVYR in [1,2]")
print(df_vars.shape, df_nhanes.shape)

# %%
keep_vars = df_vars.groupby("var").agg({"series": "count"})['series']
keep_vars = keep_vars[keep_vars == 2].index
df_vars = df_vars.query("var in @keep_vars")
df_nhanes = df_nhanes.loc[:, df_nhanes.columns.isin(keep_vars)]
print(df_vars.shape, df_nhanes.shape)

# %% [markdown]
"""
### Get rid of full NA columns
"""

# %%
drop_vars = df_nhanes.isna().sum(0) == df_nhanes.shape[0]
df_vars = df_vars.loc[~df_vars['var'].isin(drop_vars[drop_vars].index), :]
df_nhanes = df_nhanes.loc[:, df_nhanes.isna().sum(0) < df_nhanes.shape[0]]
print(df_vars.shape, df_nhanes.shape)

# %% [markdown]
"""
### Variables by type
"""

# %%
binary_vars = df_vars.loc[df_vars.is_binary == 1, 'var'].unique()
categorical_vars = ['current_past_smoking', 'occupation']
ordinal_vars = df_vars.loc[df_vars.is_ordinal == 1, 'var'].unique()
continuous_vars = df_vars.loc[~df_vars['var'].isin(
    np.concatenate((binary_vars, categorical_vars, ordinal_vars))),
                              'var']. \
    unique()

df_n_variables = pd.DataFrame.from_dict({x:{"n":len(y)} for x,y in [("binary",binary_vars),
                                                                              ("cat",categorical_vars),
                                                                              ("ordinal",ordinal_vars),
                                                                              ("cont.",continuous_vars)]},
                                        orient="index")
df_n_variables.sort_values("n")

#%% [markdown]
"""
### NA values
"""

#%%
print("Total NAs in binary columns:\n", df_nhanes.loc[:,binary_vars].isna().sum().sort_values())
print("Total NAs in cont. columns:\n", df_nhanes.loc[:,continuous_vars].isna().sum().sort_values())
print("Total NAs in categorical columns:\n", df_nhanes.loc[:,categorical_vars].isna().sum().sort_values())
print("Total NAs in ordinal columns:\n", df_nhanes.loc[:,ordinal_vars].isna().sum().sort_values())

#%%
# getting rid of mostly NA columns
df_nhanes.isna().sum().plot.hist(bins=50)
plt.show()


