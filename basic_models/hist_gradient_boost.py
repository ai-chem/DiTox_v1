from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path_to_data = "https://raw.githubusercontent.com/djzhendogg/predicting-toxicity-of-nanomaterials/master/last_df_3090.csv"
dataset = pd.read_csv(path_to_data)

dataset.drop(["Unnamed: 0", "Material", "Elements"], axis=1, inplace=True)

all_features = set(dataset.columns)
categ_features = {"Cell_type", "Coat", "Line_Primary_Cell", "Animal", "Cell_morphology", "Cell_age",
                  "Cell_organ", "Test", "Test_indicator"}

for feature in categ_features:
    dataset[feature] = dataset[feature].astype("category").cat.codes

# supporting block
print(dataset.describe())
# print(Y.describe())

out_index = dataset[dataset["Viability (%)"] == dataset["Viability (%)"].max()].index
dataset.drop(out_index, inplace=True)

# split the features and the target
label = {"Viability (%)"}
num_features = all_features - categ_features - label
X = dataset.drop("Viability (%)", axis=1)
Y = dataset["Viability (%)"]

# create bins for stratified split of the data based on the target
bins_interval = [0, 100]
bins_step = 1

bins = np.linspace(bins_interval[0],
                   bins_interval[1],
                   bins_step)

y_binned = np.digitize(Y, bins)

test_size = 0.2
random_state = 258

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    stratify=y_binned)

# assign a scaler
scaler = MinMaxScaler()

# select features with numerical values from the dataset
train_numerical_features = X_train.drop(categ_features, axis=1)
test_numerical_features = X_test.drop(categ_features, axis=1)
# get the information for scaling from train data
scaler.fit(train_numerical_features)
# scale train data
train_num_features = scaler.transform(train_numerical_features)
x_train = pd.DataFrame(train_num_features, columns=train_numerical_features.columns, index=X_train.index)
x_train = pd.merge(x_train, X_train[categ_features], right_index=True, left_index=True)

# scale test data based on the information from train data (to avoid the leakage of information about the target)
test_num_features = scaler.transform(test_numerical_features)
x_test = pd.DataFrame(test_num_features, columns=test_numerical_features.columns, index=X_test.index)
x_test = x_test.merge(X_test[categ_features], right_index=True, left_index=True)

# assign and learn the model
regressor = HistGradientBoostingRegressor()
regressor.fit(x_train, Y_train)
# estimate the model on the test data
prediction = regressor.predict(x_test)
print("Regressor R2-score: ", r2_score(Y_test, prediction))

# feature importance
# importance = regressor.feature_importances_
# features = regressor.feature_names_in_

# fig4 = plt.figure(constrained_layout=True)
# indices = np.argsort(importance)
# feature_dict = {key: value for key, value in zip(features, importance)}
# feature_dictionary = pd.DataFrame(zip(features, importance * 100), columns=["Feature ID", "Importance"])
# feature_dictionary.sort_values(by="Importance", ascending=False, inplace=True)
# feature_dictionary.to_excel("random_forest_feature_importance.xlsx")
# plt.title("HistGradientBoostingRegressor feature importance")
# sns.barplot(feature_dictionary, x="Importance", y="Feature ID")


# 10-fold cross-validation
kf = KFold(n_splits=10, random_state=random_state, shuffle=True)
kf.get_n_splits(X)

train_R2_metric_results = []
train_rmse_metric_results = []
test_R2_metric_results = []
test_rmse_metric_results = []

for train_index, test_index in kf.split(X):
    # print("Train indexes: ", train_index, "Test indexes: ", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    new_scaler = MinMaxScaler()

    train_numerical_features = X_train.drop(categ_features, axis=1)
    test_numerical_features = X_test.drop(categ_features, axis=1)

    new_scaler.fit(train_numerical_features)

    train_num_features = new_scaler.transform(train_numerical_features)
    x_train = pd.DataFrame(train_num_features, columns=train_numerical_features.columns, index=X_train.index)
    x_train = pd.merge(x_train, X_train[categ_features], right_index=True, left_index=True)

    test_num_features = new_scaler.transform(test_numerical_features)
    x_test = pd.DataFrame(test_num_features, columns=test_numerical_features.columns, index=X_test.index)
    x_test = x_test.merge(X_test[categ_features], right_index=True, left_index=True)

    new_regressor = HistGradientBoostingRegressor()
    new_regressor.fit(x_train, y_train)
    train_pred = new_regressor.predict(x_train)
    test_pred = new_regressor.predict(x_test)
    train_R2_metric_results.append(r2_score(y_train, train_pred))
    train_rmse_metric_results.append(mean_squared_error(y_train, train_pred))
    test_R2_metric_results.append(r2_score(y_test, test_pred))
    test_rmse_metric_results.append(mean_squared_error(y_test, test_pred))

# visualize the information about the model
fig5 = plt.figure()
sns.scatterplot(x=Y_test, y=prediction)
plt.title("Visualization of HistGradientBoostingRegressor prediction on test data")
plt.show()

print("leave-one-out cross-validation result (R-squared): ", r2_score(Y_test, prediction))
print("10-fold cross-validation result (R-squared): ", np.mean(test_R2_metric_results))
print("10-fold cross-validation result (RMSE): ", np.sqrt(np.mean(test_rmse_metric_results)))
