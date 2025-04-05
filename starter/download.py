import pandas as pd
import json
from ucimlrepo import fetch_ucirepo

# fetch dataset
census_income = fetch_ucirepo(id=20)

# data (as pandas dataframes)
X = census_income.data.features
y = census_income.data.targets

# metadata
print(census_income.metadata)

# variable information
print(census_income.variables)

df_features = pd.DataFrame(X)
df_target = pd.DataFrame(y)

df_features.to_csv(r"data/census_download_income_features.csv", index=False)
df_target.to_csv(r"data/census_download_income_target.csv", index=False)

with open(r"data/census_download_income_metadata.json", "w") as f:
    f.write(json.dumps(census_income.metadata))
