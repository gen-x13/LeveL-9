# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:26:18 2026

@author: GenXCode
"""

# Used for calculations
import numpy as np

# Plotting results
from matplotlib import pyplot as plt

# Creating the pipeline combining model and preprocessor
from sklearn.pipeline import make_pipeline

# Standardizing the data and Polynomial Features function before using the Model
from sklearn.preprocessing import StandardScaler

# Standardizing the data because of its dimension
from sklearn.decomposition import PCA

# Handling and creating the model used for the birdsong clustering
from sklearn.cluster import KMeans

# Splitting the data into a trainset and testset
from sklearn.model_selection import train_test_split

# Silhouette Method
from sklearn.metrics import silhouette_samples, silhouette_score

# Import handling the path to files
from pathlib import Path

# Path to store the model graph pictures
image_path = Path(__file__).parent / "pictures"

# Class that'll be called inside the web app
class BirdModel():
    def __init__(self, df, clusters):
        
        # Using the dataframe created in the bird_sound file
        self.df = df
        
        # Removing the column names, species, audio sample 
        self.df1 = self.df.drop(['Name', 'Species', 'Audio'], axis= 1)
        
        # Separating the "Name" and "Species" columns from the rest
        # They are unnecessary.
        self.X = self.df1
        
        print(f"Nombre de features: {self.X.shape[1]}")
        print(f"Nombre d'audios (samples): {self.X.shape[0]}")
        print(f"Nombre de features: {self.X.shape[1]}")
        
        print("X shape before PCA:", self.X.shape)
        
        # Number of clusters choosen by the user
        self.clusters = clusters
        
    # Function creating the pipeline and tuning it for better performances
    def preprocessor_tuning(self):
        
        # Preprocessor which will be used with all the models
        #self.preprocessor = make_pipeline(StandardScaler(), PCA())
        
        # After the transformation 
        #self.data = self.preprocessor.fit(self.X)
        
        # Number of components in pca to be plot on x-axis
        #ncomponents = len(self.preprocessor['pca'].explained_variance_ratio_.cumsum())
        
        """
        # Plotting the cumulative variance plot (looks like the elbow method)
        # A rule of thumb is to preserve around 80 % of the variance
        plt.plot(range(1, ncomponents+1), self.preprocessor['pca'].explained_variance_ratio_.cumsum(), 
                 marker='o', linestyle='--')
        plt.title("Explained Variance by components (PCA)")
        plt.xlabel('Number of components')
        plt.xticks(np.arange(1,ncomponents+1,1))
        plt.ylabel('Cumulative Explained Variance')
        
        #plt.savefig(f'{image_path}/Rule of Thumb.jpg', dpi=900, bbox_inches='tight')
        
        plt.show()
        """
       
        # Preprocessor which will be used with all the models
        self.new_preprocessor = make_pipeline(StandardScaler(), PCA(3))
        
        # New fit transform on X to test this data with the Elbow Method
        self.new_data = self.new_preprocessor.fit_transform(self.X)
        
        # Elbow Method
        #self.elbow_method(self.new_data) # k = 3
        
        # Silhouette Method
        #self.silhouette_method(self.new_data) # k = 2 or 3
        
        # The model before tuning the hyperparameters
        self.model_b = make_pipeline(StandardScaler(),
                                     PCA(3),
                                     KMeans(n_clusters=self.clusters, init="k-means++"))
                                    # KMeans(n_clusters=3)
        
        # Calling the evaluation function and
        # Taking the results : df and the pca components for the interface
        
        df_result, pca_data = self.model_prediction(self.new_data, self.model_b)
        
        return df_result, pca_data
        
    # Silhouette Method Iteration
    def silhouette_method(self, X):
        
        # Silhouette Metric
        
        sil_scores = list()
        for i in range(2, 25):
            kmeans = KMeans(n_clusters = i, random_state = 42)
            kmeans.fit(X)
            sil_scores.append(silhouette_score(X, kmeans.labels_))
        """
        plt.plot(range(2, 25), sil_scores, color = 'salmon')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette score')
        plt.title('Silouette metrics')
        plt.axvline(x = sil_scores.index(max(sil_scores))+2, linestyle = 'dotted', color = 'red') 
        
        plt.savefig(f'{image_path}/Silhouette_metrics.jpg', dpi=900, bbox_inches='tight')
        
        plt.show()
        """
        
        # Silhouette Method
        """
        # Checking silhouette coefficient
        for i,k in enumerate([2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
            
            fig, ax = plt.subplots(1,2,figsize=(15,5))
            
            # Run the kmeans algorithm
            km = KMeans(n_clusters=k)
            self.X_predict = km.fit_predict(self.new_data)
            centroids  = km.cluster_centers_
            
            # Getting silhouette
            silhouette_vals = silhouette_samples(self.X, self.X_predict)
            
            # Get the average silhouette score 
            avg_score = np.mean(silhouette_vals)
        
            print(
                "For n_clusters =",
                k,
                "The average silhouette_score is :",
                silhouette_vals,
            )
        
            # Plotting silhouette
            X_lower = 0
            
            for cluster in np.unique(self.X_predict):
                
               self.cluster_silhouette_vals = silhouette_vals[self.X_predict == cluster]
               self.cluster_silhouette_vals.sort()
               X_upper = X_lower + len(self.cluster_silhouette_vals)
               
               ax[0].barh(range(X_lower, X_upper), self.cluster_silhouette_vals)
               ax[0].text(-0.03,(X_lower + X_upper)/2,str(cluster))
               X_upper = X_lower 
                   
            ax[0].axvline(avg_score,linestyle ='--',
            linewidth =2,color = 'green')
            ax[0].set_yticks([])
            ax[0].set_xlim([-0.1, 1])
            ax[0].set_xlabel('Silhouette coefficient values')
            ax[0].set_ylabel('Cluster labels')
            ax[0].set_title('Silhouette plot for the various clusters');
             
            # scatter plot of data colored with labels
            
            ax[1].scatter(self.new_data[:, 0], self.new_data[:, 1], 
                          # PC1 = First col  ; PC2 = Second col
                          c = self.X_predict)
            ax[1].scatter(centroids[:,0],centroids[:,1],
            marker = '*' , c= 'r',s =250);
            ax[1].set_xlabel('Eruption time in mins')
            ax[1].set_ylabel('Waiting time to next eruption')
            ax[1].set_title('Visualization of clustered data', y=1.02)
            
            plt.tight_layout()
            plt.suptitle(f' Silhouette analysis using k = {k}', fontsize=16,
                         fontweight = 'semibold')
            
            plt.savefig(f'{image_path}/Silhouette_analysis_{k}.jpg', dpi=900)
            
            plt.show()
           """     
    # Searching the Elbow Method to be used as k clusters with K-Means
    def elbow_method(self, data):
        
        wcss = [] # sum of squares of distances of datapoints
        for i in range(1,21):
           kmeans_pca = KMeans(n_clusters = i, init = "k-means++")
           kmeans_pca.fit(data)
           wcss.append(kmeans_pca.inertia_)
           
        """
        # Visualisation du "Elbow Method"
        plt.figure(figsize=(10,10))
        plt.plot(range(1,21), wcss, marker='o', linestyle='--')
        plt.xlabel('Number of clusters')
        plt.xticks(np.arange(0,22,1))
        plt.ylabel('WCSS')
        plt.title("Elbow Method")
        
        plt.savefig(f'{image_path}/Elbow Method.jpg', dpi=900, bbox_inches='tight')
        
        plt.show()
        """
    
    # Function evaluating the model and 
    # version of the prediction deployed and used
    def model_prediction(self, X, model):
        
        # Training and Prediction
        label = model.fit_predict(X)
        
        # Creating Labels to plot KMeans results
        label_one = X[label == 0]
        label_two = X[label == 1]
        label_three = X[label == 2]
        label_four = X[label == 3]
        label_five = X[label == 4]
        label_six = X[label == 5]
        label_seven = X[label == 6]
        label_eight = X[label == 7]
        label_nine = X[label == 8]
        
        # Visualizing the clusters with each labels/colors
        """
        plt.scatter(label_one[:,0], label_one[:,1], color = 'red')
        plt.scatter(label_two[:,0], label_two[:,1], color = 'black')
        plt.scatter(label_three[:,0], label_three[:,1], color = 'blue')
        plt.scatter(label_four[:,0], label_four[:,1], color = 'turquoise')
        plt.scatter(label_five[:,0], label_five[:,1], color = 'royalblue')
        plt.scatter(label_six[:,0], label_six[:,1], color = 'indigo')
        plt.scatter(label_seven[:,0], label_seven[:,1], color = 'crimson')
        plt.scatter(label_eight[:,0], label_eight[:,1], color = 'coral')
        plt.scatter(label_nine[:,0], label_nine[:,1], color = 'green')
        
        plt.title("Data After Clustering")
        
        plt.savefig(f'{image_path}/After Clustering (3).jpg', dpi=900, bbox_inches='tight')
        
        plt.show()
        """
        
        # Adding a column for the clusters
        self.df['Clusters'] = label
        
        # Return it to use it with the interface
        return self.df, X
 
        
    # Running the program through the bird sound program    
    def run(self):
        
        # Calling the function and starting the model process
        return self.preprocessor_tuning()
     