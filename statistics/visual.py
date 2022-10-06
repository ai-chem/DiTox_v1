import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sunbird.categorical_encoding import frequency_encoding

path_to_data = "https://raw.githubusercontent.com/djzhendogg/predicting-toxicity-of-nanomaterials/master/last_df_3090.csv"
dataset = pd.read_csv(path_to_data)

dataset.drop(dataset[dataset["Viability (%)"] < 0].index, inplace=True)
print(dataset["Viability (%)"].describe())

dataset.drop("Unnamed: 0", axis=1, inplace=True)
# visualization of the data
metal_set = {"Ag", "Al", "Au", "Cd", "Ce", "Co", "Cu", "Fe", "Mg", "Mn", "Ni", "Pt", "Ti", "Zn", "Zr"}
nonmetal_set = set(dataset.columns) - metal_set

oxide_set = {"Al2O3", "CdO", "CeO2", "Co3O4", "Cu2O", "CuO", "Fe3O4", "MgO", "MnO", "NiO", "SiO2", "TiO2", "ZnO",
             "ZrO2"}
non_oxide_set = set(dataset.columns) - oxide_set

support_ds = dataset.copy()

metal_data = None
for value in metal_set:
    if metal_data is None:
        metal_data = support_ds[support_ds["Elements"] == value]
    else:
        metal_data = pd.concat([metal_data, support_ds[support_ds["Elements"] == value]], axis=0)

support_metal_data = metal_data.replace(metal_data["Elements"], 1)

metal_y = metal_data["Viability (%)"]
nonmetal_data = support_ds.drop(metal_data.index)
support_nonmetal_data = nonmetal_data.replace(nonmetal_data["Elements"], 2)
nonmetal_y = nonmetal_data["Viability (%)"]

oxide_data = None
for value in oxide_set:
    if oxide_data is None:
        oxide_data = support_ds[support_ds["Material"] == value]
    else:
        oxide_data = pd.concat([oxide_data, support_ds[support_ds["Material"] == value]], axis=0)

non_oxide_data = support_ds.drop(oxide_data.index)

metal_plot = plt.figure()
plt.plot(metal_data["Elements"], metal_y, "bo", label="Metalic particles")
plt.plot(nonmetal_data["Elements"], nonmetal_y, "ro", label="Non-metalic particles")
plt.title("Dependency of the viability on the type of the material (metalic / non-metalic)")
plt.legend(loc="upper right")

metal_plot2 = plt.figure(constrained_layout=True)
plt.plot(oxide_data["Material"], oxide_data["Viability (%)"], "bo", label="Oxide particles")
plt.plot(non_oxide_data["Material"], non_oxide_data["Viability (%)"], "ro", label="Non-oxide particles")
plt.title("Dependency of the viability on the type of the material (oxide / not oxide)")
plt.legend(loc="upper right")
plt.xticks(rotation=45)
plt.xlabel("Material type")
plt.ylabel("Viability (%)")

# TSNE embedding
all_features = set(dataset.columns)
categ_features = {"Cell_type", "Coat", "Line_Primary_Cell", "Animal", "Cell_morphology", "Cell_age",
                  "Cell_organ", "Test", "Test_indicator", "Elements", "Material"}

for feature in categ_features:
    dataset[feature] = dataset[feature].astype("category").cat.codes

split_ds = None
# 5 quantiles for viability
quantile_values = [np.quantile(dataset["Viability (%)"].tolist(), quant / 10) for quant in range(5)]
col_palette = sns.color_palette(n_colors=5)
target = "Viability (%)"
all_docs_number = []
viability_plot1 = plt.figure()
for number in range(len(quantile_values)):
    # if the value is in the middle part of dataset
    # we need to consider left and right borders of bins
    if number < len(quantile_values) - 1:
        left_border = quantile_values[number]
        right_border = quantile_values[number + 1]
        doc_bin = dataset[dataset[target] >= left_border]
        doc_bin.drop(doc_bin[doc_bin[target] >= right_border].index, inplace=True)
    # if the values are in the end of the dataset
    # we consider only the left border
    else:
        left_border = quantile_values[number]
        doc_bin = dataset[dataset[target] >= left_border]
    # encode the target value with the number of bin
    # to avoid target information leakage
    doc_bin[target] = number
    all_docs_number.append(doc_bin[target].count())
    # fill the supporting dataset with created bins
    if split_ds is None:
        split_ds = pd.DataFrame(doc_bin, columns=dataset.columns)
    else:
        split_ds = pd.concat([split_ds, doc_bin])

X = split_ds.drop("Viability (%)", axis=1)
Y = split_ds["Viability (%)"]

emb = TSNE(learning_rate="auto", perplexity=35)
x = emb.fit_transform(X, y=Y)
emb_ds = pd.DataFrame(x, index=dataset.index, columns=["Component_1", "Component_2"])
emb_ds = emb_ds.merge(Y, right_index=True, left_index=True)
emb_ds.rename(columns={"Viability (%)": "Viability quantiles"}, inplace=True)
sns.scatterplot(data=emb_ds, x="Component_1", y="Component_2", hue="Viability quantiles", palette="colorblind",
                legend="full", s=100)
# plt.show()

print("Quantiles for viability: ", quantile_values)

z = linkage(dataset, method="ward")
# dend = dendrogram(z)
# plt.show()

num_ds = dataset.drop(categ_features, axis=1)
cat_ds = dataset[categ_features]

for feature in categ_features:
    frequency_encoding(cat_ds, feature)

# enc_ds = frequency_encoding(dataset, categ_features)
# sns.clustermap(num_ds, dendrogram_ratio=(.1, .2), cbar_pos=(0, .2, .03, .4), cmap="vlag", metric="correlation")
sns.clustermap(z)
plt.show()
