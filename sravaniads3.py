import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from scipy import stats

from scipy.optimize import curve_fit

import seaborn as sns


def data_read(filename):
    """

This function reads data from a CSV file and returns two cleaned dataframes. One dataframe has years as columns and countries and indicators as rows, while the other has countries as columns and years and indicators as rows.

Parameters:

filename (str): Name of the CSV file to read data from.

Returns:

df_years (pandas.DataFrame): Dataframe with years as columns and countries and indicators as rows.

df_countries (pandas.DataFrame): Dataframe with countries as columns and years and indicators as rows.

"""

    # read the CSV file and skip the first 4 rows

    df = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns

    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']

    df = df.drop(cols_to_drop, axis=1)

    # rename remaining columns

    df = df.rename(columns={'Country Name': 'Country'})

    # melt the dataframe to convert years to a single column

    df = df.melt(id_vars=['Country', 'Indicator Name'],

                 var_name='Year', value_name='Value')

    # convert year column to integer and value column to float

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # separate dataframes with years and countries as columns

    df_years = df.pivot_table(

        index=['Country', 'Indicator Name'], columns='Year', values='Value')

    df_countries = df.pivot_table(

        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # clean the data

    df_years = df_years.dropna(how='all', axis=1)

    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries


def seleted_data(df_years, countries, indicators):
    """

   This function filters the data to include only the countries, indicators, and years selected by the user, spanning from 1990 to 2020. The filtered data is then returned as a new DataFrame

    """

    years = list(range(1990, 2015))

    df = df_years.loc[(countries, indicators), years]

    df = df.transpose()

    return df


def heat_map(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of 

    columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot (in inch)



    The function does not have a plt.show() at the end so that the user 

    can save the figure.

    """

    import matplotlib.pyplot as plt  # ensure pyplot imported

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    im = ax.matshow(corr, cmap='spring')

    # setting ticks to column names

    ax.set_xticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=90)

    ax.set_yticks(range(len(corr.columns)))

    ax.set_yticklabels(corr.columns)

    # add colorbar

    cbar = fig.colorbar(im)

    # add title and adjust layout

    ax.set_title('Correlation Heatmap')

    plt.tight_layout()


def normalize_data(df):
    """

    Normalizes the data using StandardScaler.

    Parameters:

    df (pandas.DataFrame): Dataframe to be normalized.

    Returns:

    df_normalized (pandas.DataFrame): Normalized dataframe.

    """

    scaler = StandardScaler()

    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_normalized


def perform_kmeans_clustering(df, num_clusters):
    """

    Performs k-means clustering on the given dataframe.



    Args:

    data (pandas.DataFrame): Dataframe to be clustered.

    num_clusters (int): Number of clusters to form.



    Returns:

    cluster_labels (numpy.ndarray): Array of cluster labels for each data point.

    """

    # Create a KMeans instance with the specified number of clusters

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Fit the model and predict the cluster labels for each data point

    cluster_labels = kmeans.fit_predict(df)

    return cluster_labels


def plot_clustered_data(df, cluster_labels, cluster_centers):
    """

    Plots the data points and cluster centers.



    Args:

    data (pandas.DataFrame): Dataframe containing the data points.

    cluster_labels (numpy.ndarray): Array of cluster labels for each data point.

    cluster_centers (numpy.ndarray): Array of cluster centers.

    """

    # Set the style of the plot

    plt.style.use('seaborn')

    # Create a scatter plot of the data points, colored by cluster label

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1],

                         c=cluster_labels, cmap='rocket')

    # Plot the cluster centers as black X's

    ax.scatter(cluster_centers[:, 0], cluster_centers[:,

               1], s=200, marker='X', c='black')

    # Set the x and y axis labels and title

    ax.set_xlabel(df.columns[0], fontsize=12)

    ax.set_ylabel(df.columns[1], fontsize=12)

    ax.set_title("K-Means Clustering Results", fontsize=14)

    # Add a grid and colorbar to the plot

    ax.grid(True)

    plt.colorbar(scatter)

    # Show the plot

    plt.show()


def filter_population_data(filename, countries, indicators, start_year, end_year):
    """

    Reads a CSV file containing population data, filters it by countries and indicators, and returns a dataframe with

    years as columns and countries and indicators as rows.

    Parameters:

    filename (str): Path to the CSV file.

    countries (list): List of country names to filter by.

    indicators (list): List of indicator names to filter by.

    start_year (int): Starting year to select data from.

    end_year (int): Ending year to select data from.

    Returns:

    population_data (pandas.DataFrame): Filtered and pivoted population data.

    """

    # read the CSV file and skip the first 4 rows

    population_data = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns

    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']

    population_data = population_data.drop(cols_to_drop, axis=1)

    # rename remaining columns

    population_data = population_data.rename(

        columns={'Country Name': 'Country'})

    # filter data by selected countries and indicators

    population_data = population_data[population_data['Country'].isin(countries) &

                                      population_data['Indicator Name'].isin(indicators)]

    # melt the dataframe to convert years to a single column

    population_data = population_data.melt(id_vars=['Country', 'Indicator Name'],

                                           var_name='Year', value_name='Value')

    # convert year column to integer and value column to float

    population_data['Year'] = pd.to_numeric(

        population_data['Year'], errors='coerce')

    population_data['Value'] = pd.to_numeric(

        population_data['Value'], errors='coerce')

    # pivot the dataframe to create a single dataframe with years as columns and countries and indicators as rows

    population_data = population_data.pivot_table(index=['Country', 'Indicator Name'],

                                                  columns='Year', values='Value')

    # select specific years

    population_data = population_data.loc[:, start_year:end_year]

    return population_data


def exp_growth(x, a, b):

    return a * np.exp(b * x)


def err_ranges(xdata, ydata, popt, pcov, alpha=0.05):

    n = len(ydata)

    m = len(popt)

    df = max(0, n - m)

    tval = -1 * stats.t.ppf(alpha / 2, df)

    residuals = ydata - exp_growth(xdata, *popt)

    stdev = np.sqrt(np.sum(residuals**2) / df)

    ci = tval * stdev * np.sqrt(1 + np.diag(pcov))

    return ci


def predict_future(population_data, countries, indicators, start_year, end_year):

    # select data for the given countries, indicators, and years

    data = filter_population_data(population_data, countries,

                                  indicators, start_year, end_year)

    # calculate the growth rate for each country and year

    growth_rate = np.zeros(data.shape)

    for i in range(data.shape[0]):

        popt, pcov = curve_fit(

            exp_growth, np.arange(data.shape[1]), data.iloc[i])

        ci = err_ranges(np.arange(data.shape[1]), data.iloc[i], popt, pcov)

        growth_rate[i] = popt[1]

    # plot the growth rate for each country

    fig, ax = plt.subplots()

    for i in range(data.shape[0]):

        ax.plot(np.arange(data.shape[1]), data.iloc[i],

                label=data.index.get_level_values('Country')[i])

    ax.set_xlabel('Year')

    ax.set_ylabel('Indicator Value')

    ax.set_title(', '.join(indicators))

    ax.legend(loc='best')

    plt.show()


if __name__ == '__main__':

    # Read the data

    df_years, df_countries = data_read(

        r"C:\Users\shrav\Downloads\ads3\worldbankdata.csv")

    # subset the data for the indicators of interest and the selected countries

    indicators = ['Population, total', 'Urban population']

    countries = ['United States', 'Italy', 'China', 'Japan']

    df = seleted_data(df_years, countries, indicators)

    # normalize the data

    df_normalized = normalize_data(df)

    # perform clustering

    n_clusters = 3

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    cluster_labels = kmeans.fit_predict(df_normalized)

    cluster_centers = kmeans.cluster_centers_

print("Clustering Results points cluster_centers")

print(cluster_centers)

# plot the results

plot_clustered_data(df_normalized, cluster_labels, cluster_centers)


predict_future(r"C:\Users\shrav\Downloads\ads3\worldbankdata.csv", [

               'Italy', 'China', 'United States'], ['Population growth (annual %)'], 1990, 2020)

heat_map(df, size=8)
