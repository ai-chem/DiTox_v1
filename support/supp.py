import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

path = "https://raw.githubusercontent.com/djzhendogg/predicting-toxicity-of-nanomaterials/master/last_df_3090.csv"
ds = pd.read_csv(path)
dataset = ds.copy()

dataset.drop("Unnamed: 0", axis=1, inplace=True)

categ_features = ["Cell_type", "Coat", "Line_Primary_Cell", "Animal", "Cell_morphology", "Cell_age",
                  "Cell_organ", "Test", "Test_indicator", "Elements", "Material"]

for feature in categ_features:
    dataset[feature] = dataset[feature].astype('category').cat.codes




train_dataset = dataset.copy()


# _ = sns.pairplot(
#     train_dataset,
#     x_vars=train_dataset.drop('Viability (%)', axis=1).columns, y_vars='Viability (%)',
#     kind='reg', diag_kind='kde', plot_kws={'scatter_kws': {'alpha': 0.1}})

# plt.show()

fig2, ax_ = plt.subplots(5, 4, figsize=(12, 11))
ax = []
for i in ax_:
    ax += i.tolist()
for number, column in enumerate(dataset.drop("Viability (%)", axis=1).columns):
    ax[number] = sns.pairplot(data=dataset, x_vars=column, y_vars="Viability (%)",)# ax=ax[number])
fig2.suptitle("Violin plots for columns in db")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)
plt.show()