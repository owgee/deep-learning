import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

HW1_Data = pd.read_csv('HW1_Data.csv')

# 1. Data Exploration
# Display basic information about the dataset in order to understand it
print("Data Information:")
print(HW1_Data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(HW1_Data.describe())
print("\nFirst Few Rows:")
print(HW1_Data.head())

# Check for any missing values - the result shows we don't have any
print("\nMissing Values:")
print(HW1_Data.isnull().sum())

# Check unique values in categorical columns [we know these are categorical by reading from the data description],
# and they're already encoded with 0 and 1 so no further manipulation is required
categorical_columns = ['webcap', 'marryyes', 'travel', 'pcown', 'creditcd']
for col in categorical_columns:
    unique_values = HW1_Data[col].unique()
    print(f"\nUnique values in {col}:")
    print(unique_values)

# Select numerical features for scatterplot analysis (still part of data exploration)
numerical_features = ['revenue', 'outcalls', 'incalls', 'eqpdays','months', 'retcalls']

# histograms for numerical variables
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=HW1_Data, x=feature, kde=True)
    plt.title(f'Histogram of {feature}')
    plt.show()

# Correlation heatmap for numerical features (to understand the correlation between features)
correlation_matrix = HW1_Data[numerical_features].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numerical Features')
plt.show()

# Pair plot for numerical features - this helps with general perception of data and looking for possible outliers
sns.pairplot(data=HW1_Data[numerical_features], diag_kind='kde')
plt.suptitle('Pair Plot for Numerical Features')
plt.show()

# Regression lines of fit
target_variable = 'churndep'
sns.set(style="whitegrid")

# Create subplots for each feature
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
fig.subplots_adjust(hspace=0.5)

for i, feature in enumerate(numerical_features):
    row = i // 3
    col = i % 3
    ax = axes[row][col]

    # Create a scatter plot
    sns.scatterplot(data=HW1_Data, x=feature, y=target_variable, ax=ax)

    # Fit a regression line using seaborn's regplot
    sns.regplot(data=HW1_Data, x=feature, y=target_variable, ax=ax, color='red')

    ax.set_title(f'Regression Line for {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel(target_variable)

# Show the plots
plt.tight_layout()
plt.show()

# 2. Splitting the data -------------------------------------------------------

# Split the data into features (X) and the target variable (y)
X = HW1_Data.drop('churndep', axis=1)  # Features
y = HW1_Data['churndep']  # Target variable

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# 3. Decision Tree -------------------------------------------------------------
# Create a decision tree classifier with entropy and no maximum depth
dt_classifier = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Because our data is very right skewed and also features' scales are very different, we standardize them
# (this will prove an increase in accuracy of our decision tree)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = X_train_scaled
X_test = X_test_scaled
# Fit the model on the training data
dt_classifier.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(14, 6), dpi=400)
plot_tree(dt_classifier, filled=True, class_names=['No Churn', 'Churn'])
plt.title(f'Decision Tree with Criterion: Entropy and no maximum depth!')
plt.show()

# Make predictions on the test set using the best model
y_pred = dt_classifier.predict(X_test)

# Evaluate the best model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the performance metrics
print(f'Accuracy of the entropy Model with no maximum depth: {accuracy}')
print(f'Confusion Matrix:\n{confusion_mat}')
print(f'Classification Report:\n{classification_rep}')

# 4. Other parameters exploration -----------------------------------------------

# Define a range of different parameters to explore: criterion (we try both entropy
# and gini to determine which one is the best together with max_depth, min_samples_split and min_samples_leaf )
param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Create a decision tree classifier with the best hyperparameters
best_dt_classifier = DecisionTreeClassifier(**best_params, random_state=42)

# Fit the best model on the training data
best_dt_classifier.fit(X_train, y_train)

# Make predictions on the test set using the best model
y_pred = best_dt_classifier.predict(X_test)

# Evaluate the best model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Plot the decision tree of the best model
plt.figure(figsize=(14, 6), dpi=400)
plot_tree(best_dt_classifier, filled=True, class_names=['No Churn', 'Churn'])
plt.title(f'Decision Tree with the best hyperparameters: {best_params}')
plt.show()

# Print the best hyperparameters and other performance metrics
print(f'Best Hyperparameters: {best_params}')
print(f'Accuracy of Best Model: {accuracy}')
print(f'Confusion Matrix:\n{confusion_mat}')
print(f'Classification Report:\n{classification_rep}')
# ---------------------------------------------------------



