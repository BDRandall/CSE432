import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns



# Load the data
data = pd.read_excel('../data/2023/encoded_allstats.xlsx')

# Columns to remove that might contain outcome-related information
necessary = ['+/-', 'OFFRTG', 'DEFRTG', 'NETRTG', 'PIE', 'PTS', 'FGM', 'FTM', '3PM', 'FGA', 'FTA', '3PA', 'TEAM_encoded', 'OPPONENT_encoded', 'AST.1']

# columns_to_remove = necessary + ['TEAM_encoded', 'OPPONENT_encoded', 'TS%', 'EFG%', 'REB%', 'FG%', 'AST.1', 'AST', 'DREB', 'REB', '3P%', 'AST/TO']
columns_to_remove = necessary
# Drop these columns from the dataset along with the target variable 'W/L'
X_filtered = data.drop(columns_to_remove + ['W/L'], axis=1)
y = data['W/L']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



correlation_matrix = pd.DataFrame(X_train_scaled, columns=X_train.columns).corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, cbar_kws={"shrink": .5})

# Adding title and labels
plt.title('Correlation Matrix Heatmap')
plt.xlabel('Predictor Variables')
plt.ylabel('Predictor Variables')

# Save the figure if you want to use it in a report or presentation
plt.savefig('correlation_matrix_heatmap.png')

# Show plot
# plt.show()

def logisitic_reg():

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Predict the test set results and evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("LOGISITC REGRESSION")
    print(f'Accuracy: {accuracy}')
    # print('Classification Report:')
    # print(report)
    

    feature_names = X_filtered.columns.tolist()

    coefficients = model.coef_[0]
    coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # Sort the DataFrame based on the absolute value of the coefficients for better visualization
    coeff_df = coeff_df.reindex(coeff_df.Coefficient.abs().sort_values(ascending=False).index)

    # Increase the size for readability
    plt.figure(figsize=(10, 10))

    # Create a more contrasting color palette
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Generate the heatmap
    sns.heatmap(coeff_df.set_index('Feature'), annot=True, fmt=".2f", linewidths=.5, cmap=cmap)

    # Improve the readability of the y-axis labels
    plt.yticks(rotation=0)  # Rotate the labels to be horizontal
    plt.title('Heatmap of Coefficients in Logistic Regression')

    # Save the improved heatmap
    plt.savefig('improved_heatmap.png')
    plt.close()  # Close the figure to free up memory


def ada_boost():
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, algorithm="SAMME", random_state=42)
    ada.fit(X_train_scaled, y_train)
    # Predict the test set results and evaluate the model
    y_pred = ada.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("ADA BOOST")
    print(f'Accuracy: {accuracy}')
    # print('Classification Report:')
    # print(report)


def grad_boost():
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
    gb.fit(X_train_scaled, y_train)
    # Predict the test set results and evaluate the model
    y_pred = gb.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("GRADIENT BOOST")
    print(f'Accuracy: {accuracy}')
    # print('Classification Report:')
    # print(report)



logisitic_reg()
ada_boost()
grad_boost()