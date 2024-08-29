import pandas as pd
import numpy as np

# Import data
data = pd.read_excel(r"C:\Users\agjur\OneDrive\Desktop\Data.xlsx")
print(data.head(10))

########################### Data preparation ################################
# 1. Categorical Data

def calculate_woe(df, feature, target):
    eps = 0.000001  # Dodaje se da bi se izbeglo deljenje sa nulom
    df_woe = pd.DataFrame()
    df_woe['feature'] = df[feature]
    df_woe['target'] = df[target]

    grouped = df_woe.groupby('feature')['target'].agg(['sum', 'count'])
    grouped['non_event'] = grouped['count'] - grouped['sum']

    grouped['event_rate'] = grouped['sum'] / (grouped['sum'].sum() + eps)
    grouped['non_event_rate'] = grouped['non_event'] / (grouped['non_event'].sum() + eps)

    grouped['woe'] = np.log((grouped['event_rate'] + eps) / (grouped['non_event_rate'] + eps))

    return grouped['woe'].to_dict()
def replace_with_woe(df, categorical_columns, target):
    for feature in categorical_columns:
        woe_map = calculate_woe(df, feature, target)
        df[feature] = df[feature].map(woe_map)
    return df


categorical_columns = ['BracniStatus' , 'Obrazovanje','TipZaposlenja', 'Pol']

# Replace the categorical columns with WoE values:
data = replace_with_woe(data, categorical_columns, 'target')

print(data)

# 2. Dates

date_columns = ['Na adresi zivi od', 'Datum prvog zaposlenja', 'Datum umaticenja klijenata']

for col in date_columns:

    data[col] = pd.to_datetime(data[col], errors='coerce')


    median_date = data[col].dropna().median()

    data[col].fillna(median_date, inplace=True)

    today = pd.to_datetime(pd.Timestamp.now().date())


    def months_since(date):
        return (today.year - date.year) * 12 + today.month - date.month


    data[f'Broj meseci od {col}'] = data[col].apply(months_since)


# Univariate analysis (numeric features)

cols = ['Godine','Broj godina u banci',
'Stanje na racunu (EUR)','Broj meseci od Na adresi zivi od','Broj meseci od Datum prvog zaposlenja'	,'Broj meseci od Datum umaticenja klijenata']

# Basic Statistics for All Numerical Columns
print(data[cols].describe())

# Detection and Replacement of Outliers
def replace_outliers(df, columns, method='IQR'):
    for col in columns:
        if method == 'IQR':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR


            df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), np.nan, df[col])
        elif method == 'mean':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std


            df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), np.nan, df[col])


        df[col].fillna(df[col].median(), inplace=True)

    return df

data_final = replace_outliers(data, cols)
print(data_final[cols].describe())

#export data

data.to_excel(r'C:\Users\agjur\OneDrive\Desktop\Provera.xlsx', index=False)
