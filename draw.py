import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# First dataset
data1 = {
    'X': [80.8, 79.8, 77.1, 75.2, 71.2],
    'Y': [-2.819273743323411, -2.819273199353899, -2.7692736921514454, -2.729273458669069, -2.65]
}

# Second dataset
data2 = {
    'X': [67.13, 70.30, 74.37, 76.47, 79.05],
    'Y': [-2.146, -2.354, -2.566, -2.6779, -2.732]
}

# Create DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Create scatter plot
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(data=df1, x='Y', y='X', color='blue', s=100, label='Canopus')
scatter = sns.scatterplot(data=df2, x='Y', y='X', color='red', s=100, label='U2PL')
scatter.set_title('Scatter Plot Comparison')
scatter.set_xlabel('Uniformity')
scatter.set_ylabel('mIOU')
plt.legend(title='Dataset')
plt.savefig('scatter_plot.png')
plt.show()
