from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path_to_data = "https://raw.githubusercontent.com/djzhendogg/predicting-toxicity-of-nanomaterials/master/with_z_3090.csv"

dataset = pd.read_csv(path_to_data)

dataset.drop(["Unnamed: 0", "Material", "Test_indicator"], axis=1, inplace=True)
# dataset.drop("Material", axis=1, inplace=True)
# dataset.drop("Test_indicator", axis=1, inplace=True)

all_features = set(dataset.columns)
categ_features = {"Cell_type", "Coat", "Line_Primary_Cell", "Animal", "Cell_morphology", "Cell_age",
                  "Cell_organ", "Test", "Elements"}  # dropped "Test_indicator" to check

for feature in categ_features:
    dataset[feature] = dataset[feature].astype("category").cat.codes

print("Statistics for the dataset: \n", dataset.describe())

correlation = dataset.corr()
fig1 = plt.figure(figsize=(10, 6), constrained_layout=True)
plt.title("Correlation matrix")
sns.heatmap(correlation)
# violin plots
fig2, ax_ = plt.subplots(5, 4, figsize=(12, 11))
ax = []
for i in ax_:
    ax += i.tolist()
for number, column in enumerate(dataset.columns):
    sns.violinplot(data=dataset, x=column, ax=ax[number])
fig2.suptitle("Violin plots for columns in db")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)

# found outlying data in viability
print("Outlying viability: \n", dataset["Viability (%)"][dataset["Viability (%)"] > 500])
out_index = dataset[dataset["Viability (%)"] == dataset["Viability (%)"].max()].index
dataset.drop(out_index, inplace=True)
# violin plot for viability
fig3 = plt.figure(figsize=(10, 6), constrained_layout=True)
sns.violinplot(data=dataset["Viability (%)"], orient="h")
plt.title("Violin plot for viability without outlier data")

# generate a new feature
support_feature = (dataset["Concentration (g/L)"] + dataset["Zeta potential (mV)"]) / (max(dataset["Concentration (g/L)"].max(), dataset["Zeta potential (mV)"].max()) \
                                                             - min(dataset["Concentration (g/L)"].min(), dataset["Zeta potential (mV)"].min()))
support_feature_series = pd.Series(support_feature, index=dataset.index, name="Generated_feature")
dataset = dataset.merge(support_feature_series, right_index=True, left_index=True)

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

# split the data on train and test datasets
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

# assign catboost Pool object
train_pool = Pool(x_train, Y_train, cat_features=list(categ_features))
test_pool = Pool(x_test, Y_test, cat_features=list(categ_features))
# assign and learn the model
regressor = CatBoostRegressor()
regressor.fit(train_pool)
# estimate the model on the test data
prediction = regressor.predict(test_pool)

# feature importance
fig4 = plt.figure(constrained_layout=True)
feature_importance = regressor.get_feature_importance(prettified=True)
sns.barplot(feature_importance, x="Importances", y="Feature Id")

# 10-fold cross-validation
kf = KFold(n_splits=10, random_state=random_state, shuffle=True)
kf.get_n_splits(X)
train_metric_results = []
test_metric_results = []
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

    train_pool = Pool(x_train, y_train, cat_features=list(categ_features))
    test_pool = Pool(x_test, y_test, cat_features=list(categ_features))

    new_regressor = CatBoostRegressor()
    new_regressor.fit(train_pool)
    train_pred = new_regressor.predict(x_train)
    test_pred = new_regressor.predict(x_test)
    train_r_2_metr = r2_score(y_train, train_pred)
    test_r_2_metr = r2_score(y_test, test_pred)
    train_metric_results.append(train_r_2_metr)
    test_metric_results.append(test_r_2_metr)

# visualize the information about the model
fig5 = plt.figure()
sns.scatterplot(x=Y_test, y=prediction)
plt.title("Visualization of CatBoostRegressor prediction on test data")
plt.show()

print("leave-one-out cross-validation result (R2-score): ", regressor.score(test_pool))
print("10-fold cross-validation result (R2-score): ", np.mean(test_metric_results))
