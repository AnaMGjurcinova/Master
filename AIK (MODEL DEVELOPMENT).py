import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

############################ MODEL DEVELOPMENT #####################

data = pd.read_excel(r"C:\Users\agjur\OneDrive\Desktop\Provera.xlsx")
print(data.info)
data = data.drop(['Na adresi zivi od',	'Datum prvog zaposlenja',	'Datum umaticenja klijenata'], axis=1)

correlation_matrix = data.corr()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation matrix')
plt.show()

# X - all the predictive features
X = data.drop(columns=['target', 'Broj godina u trenutku kupovine poslednjeg kredita'])
# Y - target variable
Y = data['target']

# Feature selection
model = LogisticRegression(max_iter=1000)

# Stepwise selection
sfs = SequentialFeatureSelector(model, k_features=5, forward=True, floating=False, scoring='accuracy', cv=5)
sfs = sfs.fit(X, Y)

# Selected features
selected_features = X.columns[list(sfs.k_feature_idx_)]
print("Selected features are:", selected_features)

X_selected = X[selected_features]
custid = data['CustID']

# VIF Calculation
X = data[selected_features]
X = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# Model Development, Testing, and Results Presentation

X_train, X_test, y_train, y_test, custid_train, custid_test= train_test_split(X_selected, Y, custid, test_size=0.2, random_state=42)

# Perform Grid search to optimize hyperparameters and improve model performance

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

model = LogisticRegression(class_weight='balanced', max_iter=2000)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Best parameters found in the search
print("Best parameters found: ", grid_search.best_params_)
print("Best F1 score found: ", grid_search.best_score_)

# Training and testing the best model
best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)

coef = best_model.coef_[0]
features = X_train.columns

# Model representation
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coef})
print(coef_df)
# Model test probability for case 1 (Target = 1)
y_proba = best_model.predict_proba(X_test)[:, 1]

new_threshold = 0.35
y_pred_custom = (y_proba > new_threshold).astype(int)

# Evaluation with standard threshold (0.5) and adjusted threshold (0.35)
print("Classification report with default threshold (0.5):")
print(classification_report(y_test, y_pred))

print("Classification report with custom threshold (0.3):")
print(classification_report(y_test, y_pred_custom))


########## Distribution of true values and accuracy estimation ###################

results_df = pd.DataFrame({
    'CustID': custid_test,
    'Probability_Class_1': y_proba,
    'True_Target': y_test
})


results_df['Probability_Class_Bin'] = pd.qcut(results_df['Probability_Class_1'], q=10, duplicates='drop')


class_distribution = results_df.groupby('Probability_Class_Bin').agg({
    'CustID': 'count',
    'Probability_Class_1': ['min', 'max', 'mean'],
    'True_Target': 'sum'
}).reset_index()


class_distribution.columns = ['Probability_Class_Bin', 'Client_Count', 'Min_Probability', 'Max_Probability', 'Mean_Probability', 'Total_Target']


class_distribution = class_distribution.sort_values(by='Min_Probability', ascending=False).reset_index(drop=True)


print(class_distribution)

color = 'green'  # You can change this to any color you prefer

# Create a density histogram using seaborn
plt.figure(figsize=(8, 6))
sns.histplot(data['Godine'], color=color, linewidth=0)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Density')
plt.title(f'Density Histogram of Age')

# Show the plot
plt.show()






