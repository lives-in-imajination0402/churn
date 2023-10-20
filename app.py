import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

# Load the dataset
data = pd.read_excel('/content/customer_churn_large_dataset.xlsx')

# Function to print dataset overview
def dataoverview(df, message):
    print(f'{message}:\n')
    print('Number of rows: ', df.shape[0])
    print("Number of features:", df.shape[1])
    print("Data Features:")
    print(df.columns.tolist())
    print("Missing values:", df.isnull().sum().values.sum())
    print("Unique values:")
    print(df.nunique())

dataoverview(data, 'Overview of the dataset')

# Visualization of Churn distribution
target_instance = data["Churn"].value_counts().to_frame()
target_instance = target_instance.reset_index()
target_instance = target_instance.rename(columns={'index': 'Category'})
fig = px.pie(target_instance, values='Churn', names='Category', color_discrete_sequence=["green", "red"],
             title='Distribution of Churn')
fig.show()

# Group the data by 'Gender' and 'Churn' and count the occurrences
grouped_data = data.groupby(['Gender', 'Churn']).size().unstack()

# Create a bar chart
ax = grouped_data.plot(kind='bar', figsize=(8, 6))

# Customize the chart
ax.set_xlabel('Gender')
ax set_ylabel('Count')
ax.set_title('Churn by Gender')

plt.xticks(rotation=0)  # Rotate x-axis labels if necessary
plt.legend(title='Churn', loc='upper right')

plt.show()

# Function to create a histogram for a feature
def hist(feature):
    group_df = data.groupby([feature, 'Churn']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=feature, y='Count', color='Churn',
                       marginal='box', title=f'Churn rate frequency to {feature} distribution',
                       color_discrete_sequence=["green", "red"])
    fig.show()

hist('Monthly_Bill')
hist('Total_Usage_GB')
hist('Subscription_Length_Months')

# Label encoding for categorical variables
categorical_columns = ['Location', 'Gender']
le = LabelEncoder()
for column in categorical_columns:
    data[column] = le.fit_transform(data[column])

# Drop irrelevant columns
data.drop(["CustomerID", 'Name'], axis=1, inplace=True)

# Correlation matrix
corr = data.corr()
fig = px.imshow(corr, width=1000, height=1000)
fig.show()

# Statistical tests for feature significance
categorical_features = ['Gender', 'Location']
numerical_features = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

for feature in categorical_features:
    # Create a contingency table for the chi-squared test
    contingency_table = pd.crosstab(data['Churn'], data[feature])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-squared test for {feature} - p-value: {p:.4f}")

for feature in numerical_features:
    churned = data[data['Churn'] == 1][feature]
    not_churned = data[data['Churn'] == 0][feature]
    t_stat, p = ttest_ind(churned, not_churned)
    print(f"T-test for {feature} - p-value: {p:.4f}")

# Feature scaling
sc = StandardScaler()
data['Subscription_Length_Months'] = sc.fit_transform(data[['Subscription_Length_Months']])
data['Monthly_Bill'] = sc.fit_transform(data[['Monthly_Bill'])
data['Total_Usage_GB'] = sc.fit_transform(data[['Total_Usage_GB']])

# Splitting the dataset
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a QDA model
qda = QuadraticDiscriminantAnalysis()

# Perform k-fold cross-validation
k = 5  # Number of folds
scores = cross_val_score(qda, X, y, cv=k, scoring='accuracy')

# Print the cross-validation scores
print(f"Cross-Validation Scores: {scores}")

# Calculate and print the average accuracy
average_accuracy = np.mean(scores)
print(f"Average Accuracy: {average_accuracy:.2f}")

# Fit the QDA model on the training data
qda.fit(X_train, y_train)

# Make predictions on the test data
y_pred = qda.predict(X_test)

# Evaluate the model on the test data
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Test Set Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# Hyperparameter tuning
# Define a dictionary of hyperparameters and their possible values to search over
param_grid = {
    'reg_param': [0.1, 0.2, 0.3, 0.4],  # Regularization parameter (0.0 means no regularization)
    'priors': [None, [0.2, 0.8], [0.5, 0.5]]  # Prior probabilities of classes
}

# Create a grid search object with cross-validation
grid_search = GridSearchCV(qda, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to your training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Make predictions using the best model on the test data
y_pred = best_model.predict(X_test)

# Evaluate the best model on the test data
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print evaluation metrics for the best model
print(f"Test Set Accuracy with Best Model: {accuracy:.2f}")
print("Confusion Matrix for Best Model:\n", conf_matrix)
print("Classification Report for Best Model:\n", classification_rep)

# Save the best model to disk
filename = 'best_qda_model.sav'
joblib.dump(best_model, filename)
