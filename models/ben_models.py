import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance



import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Load the data
data = pd.read_excel('../data/2023/encoded_allstats.xlsx')

# Columns to remove that might contain outcome-related information
necessary = ['W/L', '+/-', 'OFFRTG', 'DEFRTG', 'NETRTG', 'PIE', 'PTS', 'FGM', 'FTM', '3PM', 'FGA', 'FTA', '3PA', 'TEAM_encoded', 'OPPONENT_encoded', 'AST.1']

# Columns that are necessary + columns that cause multicollinearity
columns_to_remove = necessary + ['OREB', 'DREB', 'REB', 'TOV', 'EFG%', 'AST', 'TS%']

# columns_to_remove = necessary
# Drop these columns from the dataset along with the target variable 'W/L'
X_filtered = data.drop(columns_to_remove, axis=1)
y = data['W/L']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


############################
#### CORRELATION MATRIX ####
############################
def corr_matrix():
    
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


#########################
#### VIF CALCULATION ####
#########################
def vif():
    # Calculate VIF for each feature in the training set
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_train.columns
    vif_data['VIF'] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]

    # Display the VIF for each feature
    print("VIF for each feature:")
    print(vif_data)

    # Identify features with an infinite VIF
    infinite_vif_features = vif_data[vif_data['VIF'] == np.inf]['feature'].tolist()
    print("\nFeatures with Infinite VIF (suggesting perfect multicollinearity):")
    print(infinite_vif_features)

    # Set a VIF threshold to identify features with high multicollinearity
    vif_threshold = 10
    high_vif_features = vif_data[vif_data['VIF'] > vif_threshold]['feature'].tolist()
    print("\nFeatures with VIF greater than threshold of 10:")
    print(high_vif_features)


############################
#### MODEL CALCULATIONS ####
############################


def logisitic_reg():

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Predict the test set results and evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)
    print("LOGISITC REGRESSION")
    print(f'Accuracy: {accuracy}')
    # print('Classification Report:')
    # print(report)
    
    # model = LogisticRegression()
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
        'solver': ['liblinear', 'lbfgs']  # Optimization algorithms
    }

    # Setup the grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')

    # Perform the grid search on the scaled training data
    grid_search.fit(X_train_scaled, y_train)

    # Best parameters and best score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # Evaluate on the test set
    y_pred = grid_search.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Set Accuracy: {test_accuracy}')
    
    # Compute and display permutation importance
    result = permutation_importance(grid_search.best_estimator_, X_test_scaled, y_test, n_repeats=10, random_state=42)
    importances = result.importances_mean
    print("Feature Importances:")
    for i, imp in enumerate(importances):
        print(f"{X_train.columns[i]}: {imp:.3f}")
        
    # Visualize feature importances
    sns.barplot(x=importances, y=X_train.columns)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig('features importance 2.png')
    

    # feature_names = X_filtered.columns.tolist()

    # coefficients = model.coef_[0]
    # coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # # Sort the DataFrame based on the absolute value of the coefficients for better visualization
    # coeff_df = coeff_df.reindex(coeff_df.Coefficient.abs().sort_values(ascending=False).index)

    # # Increase the size for readability
    # plt.figure(figsize=(10, 10))

    # # Create a more contrasting color palette
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # # Generate the heatmap
    # sns.heatmap(coeff_df.set_index('Feature'), annot=True, fmt=".2f", linewidths=.5, cmap=cmap)

    # # Improve the readability of the y-axis labels
    # plt.yticks(rotation=0)  # Rotate the labels to be horizontal
    # plt.title('Heatmap of Coefficients in Logistic Regression')

    # # Save the improved heatmap
    # plt.savefig('improved_heatmap.png')
    # plt.close()  # Close the figure to free up memory


def ada_boost():
    ada = AdaBoostClassifier(algorithm="SAMME", random_state=42)
    ada.fit(X_train_scaled, y_train)
    # Predict the test set results and evaluate the model
    y_pred = ada.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)
    print("ADA BOOST")
    print(f'Accuracy: {accuracy}')
    # print('Classification Report:')
    # print(report)
    
    # ada = AdaBoostClassifier(random_state=42)
    param_grid_ada = {
        'n_estimators': [50, 100, 200],  # Number of boosting stages to perform
        'learning_rate': [0.01, 0.1, 1.0]  # Weight applied to each classifier at each boosting iteration
    }

    # Setup the grid search with cross-validation
    grid_search_ada = GridSearchCV(ada, param_grid_ada, cv=10, scoring='accuracy')

    # Perform the grid search on the scaled training data
    grid_search_ada.fit(X_train_scaled, y_train)

    # Best parameters and best score
    print("Best parameters for AdaBoost:", grid_search_ada.best_params_)
    print("Best cross-validation score for AdaBoost: {:.2f}".format(grid_search_ada.best_score_))

    # Evaluate on the test set
    y_pred_ada = grid_search_ada.predict(X_test_scaled)
    test_accuracy_ada = accuracy_score(y_test, y_pred_ada)
    print(f'Test Set Accuracy for AdaBoost: {test_accuracy_ada}')


def grad_boost():
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train_scaled, y_train)
    # Predict the test set results and evaluate the model
    y_pred = gb.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)
    print("GRADIENT BOOST")
    print(f'Accuracy: {accuracy}')
    # print('Classification Report:')
    # print(report)

    # gb = GradientBoostingClassifier(random_state=42)
    param_grid_gb = {
        'n_estimators': [100, 200, 300],  # Number of boosting stages to perform
        'learning_rate': [0.01, 0.1, 0.5],  # Shrinks the contribution of each tree
        'max_depth': [3, 5, 7]  # Maximum depth of the individual regression estimators
    }

    # Setup the grid search with cross-validation
    grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=10, scoring='accuracy')

    # Perform the grid search on the scaled training data
    grid_search_gb.fit(X_train_scaled, y_train)

    # Best parameters and best score
    print("Best parameters for Gradient Boosting:", grid_search_gb.best_params_)
    print("Best cross-validation score for Gradient Boosting: {:.2f}".format(grid_search_gb.best_score_))

    # Evaluate on the test set
    y_pred_gb = grid_search_gb.predict(X_test_scaled)
    test_accuracy_gb = accuracy_score(y_test, y_pred_gb)
    print(f'Test Set Accuracy for Gradient Boosting: {test_accuracy_gb}')


logisitic_reg()
# ada_boost()
# grad_boost()