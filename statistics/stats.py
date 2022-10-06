import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as ex


path = "https://raw.githubusercontent.com/djzhendogg/predicting-toxicity-of-nanomaterials/master/last_df_3090.csv"
ds = pd.read_csv(path)

dataset = ds.copy()

dataset.drop("Unnamed: 0", axis=1, inplace=True)

categ_features = ["Cell_type", "Coat", "Line_Primary_Cell", "Animal", "Cell_morphology", "Cell_age",
                  "Cell_organ", "Test", "Test_indicator", "Elements", "Material"]

for feature in categ_features:
    dataset[feature] = dataset[feature].astype('category').cat.codes

ds_array = dataset.to_numpy()

material_stat = ds['Material'].describe()
material_dict = dict()

for key in ds['Material'].unique():
    sup = len(ds[ds['Material'] == key].index)
    material_dict[key] = sup

pie_mat = ex.pie(values=material_dict.values(), names=material_dict.keys())
pie_mat.show()

x = dataset.drop('Viability (%)', axis=1)

kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
labels = kmeans.labels_
label_ser = pd.Series(labels, index=dataset.index, name='knn_label')
# dataset = dataset.merge(label_ser, right_index=True, left_index=True)
centroids = kmeans.cluster_centers_
centr_0 = centroids[0]
centr_1 = centroids[1]
centr_ds = pd.DataFrame(centroids)

support_ds = dataset.copy()
yes_coat_index = dataset[dataset['Coat'] > 0].index
support_ds.loc[yes_coat_index, 'Coat'] = 1

# 5 quantiles for viability
quantile_values = [np.quantile(dataset["Viability (%)"].tolist(), quant / 10) for quant in range(5)]
col_palette = sns.color_palette(n_colors=5)
target = "Viability (%)"
all_docs_number = []
viability_plot1 = plt.figure()
split_ds = None
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

split_ds.rename(columns={"Viability (%)": "Viability quantiles"}, inplace=True)

tsne = TSNE(perplexity=35, learning_rate='auto')


dec_ds = tsne.fit_transform(x)
dec_ds = pd.DataFrame(dec_ds, columns=['component_1', 'component_2'])

color_palt = sns.color_palette('tab20b_r')


sns.scatterplot(x=dec_ds['component_1'], y=dec_ds['component_2'], hue=ds['Material'], palette='tab20b_r')

fig_tsne_1= plt.figure()
sns.scatterplot(x=dec_ds['component_1'], y=dec_ds['component_2'], hue=support_ds['Coat'], palette='bright')

fig_tsne_2= plt.figure()
sns.scatterplot(x=dec_ds['component_1'], y=dec_ds['component_2'], hue=split_ds['Viability quantiles'], palette='bright')

plt.show()


tsne_3d = TSNE(n_components=3, perplexity=40, learning_rate='auto')
dec_3d = tsne_3d.fit_transform(x)
dec_3d = pd.DataFrame(dec_3d, columns=['component_1', 'component_2', 'component_3'])

fig_3d_mat_tsne = ex.scatter_3d(
    dec_3d, x='component_1', y='component_2', z='component_3',
    color=ds['Material'], labels={'color': 'Material'},
    color_discrete_sequence=ex.colors.qualitative.Alphabet
)

fig_3d_mat_tsne.show()

fig_3d_coat_tsne = ex.scatter_3d(
    dec_3d, x='component_1', y='component_2', z='component_3',
    color=support_ds['Coat'], labels={'color': 'Coat'},
    # color_discrete_sequence=ex.colors.qualitative.Alphabet
)

fig_3d_coat_tsne.show()

fig_3d_viab_tsne = ex.scatter_3d(
    dec_3d, x='component_1', y='component_2', z='component_3',
    color=split_ds['Viability quantiles'], labels={'color': 'Viability quantiles'},
    # color_discrete_sequence=ex.colors.qualitative.Alphabet
)

fig_3d_viab_tsne.show()

xticks = range(0, 3090, 400)

figa = plt.figure(figsize=(15, 10), constrained_layout=True)
z = sch.linkage(ds_array, method='ward', optimal_ordering=True)
dendr = sch.dendrogram(z)
plt.title('Dendrogram for the data')
plt.xlabel('documents')
plt.ylabel('distance')
plt.xticks(xticks)
plt.show()

label_1_index = label_ser[label_ser == 1].index
lb1_ds = dataset.loc[label_1_index]
lb0_ds = dataset.drop(label_1_index)

# violin plots
fig2, ax_ = plt.subplots(5, 4, figsize=(12, 11))
ax = []
for i in ax_:
    ax += i.tolist()
for number, column in enumerate(lb1_ds.columns):
    sns.violinplot(data=lb1_ds, x=column, ax=ax[number])
fig2.suptitle("Violin plots for data in cluster 1")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)
plt.show()

# violin plots
fig3, ax_ = plt.subplots(5, 4, figsize=(12, 11))
ax = []
for i in ax_:
    ax += i.tolist()
for number, column in enumerate(lb0_ds.columns):
    sns.violinplot(data=lb0_ds, x=column, ax=ax[number])
fig2.suptitle("Violin plots for data in cluster 0")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)
plt.show()

print(ds.loc[label_1_index].describe())