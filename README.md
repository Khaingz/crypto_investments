# crypto_investments
Challenge10

Module 10 Application
Challenge: Crypto Clustering
In this Challenge, you’ll combine your financial Python programming skills with the new unsupervised learning skills that you acquired in this module.

The CSV file provided for this challenge contains price change data of cryptocurrencies in different periods.

The steps for this challenge are broken out into the following sections:

- Import the Data 

Read the “crypto_market_data.csv” file from the Resources folder into a DataFrame, and use index_col="coin_id" to set the cryptocurrency name as the index. Review the DataFrame.

Generate the summary statistics, and use HvPlot to visualize your data to observe what your DataFrame contains.

Rewind: The Pandasdescribe()function generates summary statistics for a DataFrame.

### Import the required libraries and dependencies
import pandas as pd
import hvplot.pandas
from path import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

### Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

### Display sample data
df_market_data.head(10)

- Prepare the Data 
This section prepares the data before running the K-Means algorithm. It follows these steps:

### Use the StandardScaler module from scikit-learn to normalize the CSV file data. This will require you to utilize the fit_transform function.
scaled_data = StandardScaler().fit_transform(df_market_data)

Create a DataFrame that contains the scaled data. Be sure to set the coin_id index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.
### Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

### Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

### Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

### Display sample data
df_market_data_scaled.head()

- Find the Best Value for k Using the Original Data

In this section, you will use the elbow method to find the best value for k.

### Code the elbow method algorithm to find the best value for k. Use a range from 1 to 11.
k = list(range(1,11))
### Create an empy list to store the inertia values
inertia = []

### Create a for loop to compute the inertia with each possible value of k
### Inside the loop:
### 1. Create a KMeans model using the loop counter for the n_clusters
### 2. Fit the model to the data using `df_market_data_scaled`
### 3. Append the model.inertia_ to the inertia list 
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_market_data_scaled)
    inertia.append(model.inertia_)   
### View the inertia list
inertia

### Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    "k": k,
    "inertia": inertia
}

### Create a DataFrame with the data to plot the Elbow curve
df_elbow_data = pd.DataFrame(elbow_data)

Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.
### Plot a line chart with all the inertia values computed with 
### the different values of k to visually identify the optimal value for k.
df_elbow = df_elbow_data.hvplot.line(
    x="k", 
    y= "inertia", 
    title="Elbow Curve", 
    xticks=k
)
df_elbow

Answer the following question: 
Question: What is the best value for k?What is the best value for k?
Answer: # Based on the plot, 4 seems to be the optimal number for k's value.

- Cluster Cryptocurrencies with K-means Using the Original Data
In this section, you will use the K-Means algorithm with the best value for k found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.

### Initialize the K-Means model with four clusters using the best value for k.
model = KMeans(n_clusters=4)

### Fit the K-Means model using the original data.
model.fit(df_market_data_scaled)

### Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.
market_data_clusters = model.predict(df_market_data_scaled)

# View the resulting array of cluster values.
print(market_data_clusters)  

### Create a copy of the original data and add a new column with the predicted clusters.
df_market_data_scaled_predictions = df_market_data_scaled.copy()

### Add a new column to the DataFrame with the predicted clusters
df_market_data_scaled_predictions["market_data_clusters"] = market_data_clusters

### Display sample data
df_market_data_scaled_predictions.head()

### Create a scatter plot using hvPlot by setting x="price_change_percentage_24h" and y="price_change_percentage_7d". Color the graph points with the labels found using K-Means and add the crypto name in the hover_cols parameter to identify the cryptocurrency represented by each data point.
df_market_data = df_market_data_scaled_predictions.hvplot.scatter(
    x="price_change_percentage_24h", 
    y="price_change_percentage_7d", 
    by="market_data_clusters",
    hover_cols = ["coin_id"],
    title = "Scatter Plot by Cryptocurrency Market Data k=4"
)
df_market_data

- Optimize Clusters with Principal Component Analysis
In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.

### Create a PCA model instance and set n_components=3.
pca = PCA(n_components=3)

### Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame.
market_data_pca = pca.fit_transform(df_market_data_scaled)

### View the first five rows of the DataFrame. 
market_data_pca[:5]

### Retrieve the explained variance to determine how much information can be attributed to each principal component.
pca.explained_variance_ratio_

Answer the following question:
Question: What is the total explained variance of the three principal components?

Answer: The total explained variance of the three principal components are approximately 80% , the original data is represented by the three principal components we created. This percentage may vary between different code executions due to the randomness associated with running the model.

Create a new DataFrame with the PCA data. Be sure to set the coin_id index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.
### Creating a DataFrame with the PCA data
df_market_data_pca = pd.DataFrame(market_data_pca, columns =["PC1", "PC2", "PC3"])

### Copy the crypto names from the original data
df_market_data_pca["coin_id"] = df_market_data_pca.index  ****(Question here)df_market_data.index

### Set the coinid column as index
df_market_data_pca = df_market_data_pca.set_index("coin_id")

### Display sample data
df_market_data_pca.head()

- Find the Best Value for k Using the PCA Data
In this section, you will use the elbow method to find the best value for k using the PCA data.

Code the elbow method algorithm and use the PCA data to find the best value for k. Use a range from 1 to 11.
k = list(range(1,11))

### Create an empy list to store the inertia values
inertia = []

Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_market_data_pca)
    inertia.append(model.inertia_)
    
### View the inertia list
inertia

### Create a dictionary with the data to plot the Elbow curve
pca_elbow_data = {
    "k": k,
    "inertia": inertia
}

### Create a DataFrame with the data to plot the Elbow curve
df_pca_elbow_data = pd.DataFrame(pca_elbow_data)

### Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.
df_pca_elbow = df_pca_elbow_data.hvplot.line(
    x="k", 
    y= "inertia", 
    title="PCA Elbow Curve", 
    xticks=k
)

df_pca_elbow

Answer the following questions: 
Question: What is the best value for k when using the PCA data?

Answer: # Based on the plot, 4 seems to be the optimal number for k's value.
Question: Does it differ from the best k value found using the original data?

Answer: # There is no difference from the best k value using the original data.

- Cluster the Cryptocurrencies with K-means Using the PCA Data
In this section, you will use the PCA data and the K-Means algorithm with the best value for k found in the previous section to cluster the cryptocurrencies according to the principal components.

### Initialize the K-Means model with four clusters using the best value for k.
model = KMeans(n_clusters=4)

### Fit the K-Means model using the PCA data.
model.fit(df_market_data_pca)

### Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.
market_data_pca_clusters = model.predict(df_market_data_pca)

### View the resulting array of cluster values.
print(market_data_pca_clusters)

### Add a new column to the DataFrame with the PCA data to store the predicted clusters.
### Create a copy of the DataFrame with the PCA data
df_market_data_pca_predictions = df_market_data_pca.copy()

### Add a new column to the DataFrame with the predicted clusters
df_market_data_pca_predictions["market_data_pca_clusters"] = market_data_pca_clusters

### Display sample data
df_market_data_pca_predictions.head()  

### Create a scatter plot using hvPlot by setting x="PC1" and y="PC2". Color the graph points with the labels found using K-Means and add the crypto name in the hover_cols parameter to identify the cryptocurrency represented by each data point.

df_pca_market_data = df_market_data_pca_predictions.hvplot.scatter(
    x="PC1", 
    y="PC2", 
    by="market_data_pca_clusters",
    hover_cols = ["coin_id"],
    title = "Scatter Plot by Cryptocurrency Market Data PCA k=4"
)
df_pca_market_data

- Visualize and Compare the Results

In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

Create a composite plot using hvPlot and the plus (+) operator to contrast the Elbow Curve that you created to find the best value for k with the original and the PCA data.
### Composite plot to contrast the Elbow curves
(df_elbow + df_pca_elbow).cols(1)

Create a composite plot using hvPlot and the plus (+) operator to contrast the cryptocurrencies clusters using the original and the PCA data.
### Compoosite plot to contrast the clusters
(df_market_data+df_pca_market_data).cols(1)

Answer the following question: 

Question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

Answer: # The impact of using fewer features to cluster the data using has creative a better visualization making it easy to iterpreate our data.