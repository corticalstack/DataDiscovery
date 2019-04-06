import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_wine


class DataDiscovery:
    def __init__(self):
        self.data = None
        self.random_state = 20
        self.load_data()
        self.overview()
        self.column_unique_value_count()
        self.column_zero_count()
        self.column_null_count()
        self.row_count_by_target('BldgType')
        self.distribution_analysis()
        self.correlation_matrix()
        self.correlation_matrix_by_level('SalePrice')
        self.pair_plot()
        self.outliers()
        self.bivariate_analysis('SalePrice', 'GrLivArea')
        self.group_by_counts()

    def load_data(self):
        self.data = pd.read_csv('train.csv')

    def overview(self):
        print('\n', '_' * 40, 'Overview', '_' * 40)
        print('Row count:\t', '{}'.format(self.data.shape[0]))
        print('Column count:\t', '{}'.format(self.data.shape[1]))
        print('\n--- Column Values\n', self.data.columns.values)
        print('\n--- Info\n', self.data.info())
        print('\n--- Head\n', self.data.head())
        print('\n--- Tail\n', self.data.tail())
        print('\n--- Describe\n', self.data.describe())
        print('\n--- Describe incl 0\n', self.data.describe(include=['O']))

    def row_count_by_target(self, target):
        print('\n', '_' * 40, 'Row count by {}'.format(target), '_' * 40)
        series = self.data[target].value_counts()
        for idx, val in series.iteritems():
            print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / self.data.shape[0]) * 100)))

    def column_unique_value_count(self):
        print('\n', '_' * 40, 'Column unique value count', '_' * 40)
        for col in self.data:
            print('\t{} ({})'.format(col, len(self.data[col].unique())))

    def column_zero_count(self):
        print('\n', '_' * 40, 'Column zero count', '_' * 40)
        for col in self.data:
            count = (self.data[col] == 0).sum()
            print('\t{} {} ({:6.3f}%)'.format(col, count, (count / self.data.shape[0]) * 100))

    def column_null_count(self):
        print('\n', '_' * 40, 'Column null count', '_' * 40)
        total = self.data.isnull().sum().sort_values(ascending=False)
        percent = (self.data.isnull().sum() / self.data.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(missing_data.head(20))

    def distribution_analysis(self):
        print('\n', '_' * 40, 'Distribution Analysis of target variable SalePrice', '_' * 40)
        self.data['SalePrice'].describe()
        sns.distplot(self.data['SalePrice'])
        plt.show()
        print("Min: %f" % self.data['SalePrice'].min())
        print("Max: %f" % self.data['SalePrice'].max())
        print("Mean: %f" % self.data['SalePrice'].mean())
        print("Median: %f" % self.data['SalePrice'].median())
        print("Quantile 75: %f" % self.data['SalePrice'].quantile(.75))
        print("Skewness: %f" % self.data['SalePrice'].skew())
        print("Kurtosis: %f" % self.data['SalePrice'].kurt())

        var = 'OverallQual'
        data = pd.concat([self.data['SalePrice'], self.data[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000)
        plt.show()

    def correlation_matrix(self):
        corrmat = self.data.corr()
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True)
        plt.show()

    def correlation_matrix_by_level(self, level):
        corrmat = self.data.corr()
        k = 10
        cols = corrmat.nlargest(k, level)[level].index
        cm = np.corrcoef(self.data[cols].values.T)
        sns.set(font_scale=1.25)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                         yticklabels=cols.values, xticklabels=cols.values)
        plt.show()

    def pair_plot(self):
        sns.set()
        cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        sns.pairplot(self.data[cols], height=3)
        plt.show()

    def outliers(self):
        print('\n', '_' * 40, 'Outlier Analysis target SalePrice - Scaled with StandardScaler - 0 is mean', '_' * 40)
        scaled = StandardScaler().fit_transform(self.data['SalePrice'][:, np.newaxis])
        low_range = scaled[scaled[:, 0].argsort()][:10]
        high_range = scaled[scaled[:, 0].argsort()][-10:]
        print('outer range (low) of the distribution:')
        print(low_range)
        print('\nouter range (high) of the distribution:')
        print(high_range)

    def bivariate_analysis(self, f1, f2):
        colours = np.array(["red"])
        data = pd.concat([self.data[f1], self.data[f2]], axis=1)
        data.plot.scatter(x=f2, y=f1, ylim=(0, 800000), c=colours[0])
        plt.show()

    def group_by_counts(self):
        print('\n', '_' * 40, 'Group By Counts', '_' * 40)
        print(self.data.groupby('MSZoning').count())
        print(self.data.groupby('Utilities').count())
        print(self.data.groupby('Exterior1st').count())
        print(self.data.groupby('BsmtQual').count())
        print(self.data.groupby('BsmtCond').count())
        print(self.data.groupby('KitchenQual').count())
        print(self.data.groupby('SaleType').count())


data_discovery = DataDiscovery()