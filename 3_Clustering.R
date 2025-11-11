## CCM Machine Learning in R Workshop: K-Means Clustering


# df = prepped_heart_df_cluster (only contains numeric/continuous variables)

# look at correlations within the clustering dataset 
corrplot(cor(prepped_heart_df_cluster), tl.col = "black")

set.seed(66)

# perform k-means clustering
heart_kmeans <- kmeans(prepped_heart_df_cluster, centers = 3, nstart = 25)
heart_kmeans

# The kmeans() function returns an object of class “kmeans” with information about the partition:
#   1. cluster. Indicates the cluster to which each data point/observation is allocated.
#   2. centers. A matrix of cluster centers.
#   3. size. The number of data points in each cluster.
#   4. totss. The total sum of squares (within + between).
#   5. tot.withinss The total within-cluster sum of squares across clusters.
#   6. withinss. The within-cluster sum of squares for each cluster.

heart_kmeans$cluster
heart_kmeans$centers
heart_kmeans$size
heart_kmeans$totss
heart_kmeans$tot.withinss
heart_kmeans$withinss

# what are the characteristics of each sub-group identified through clustering?
aggregate(prepped_heart_df_cluster, by=list(cluster=heart_kmeans$cluster), mean)
## it's not easy to interpret this output because all the variables are scaled
## (scaling is necessary for k-means but makes the means less interpretable in original units)

# visualize the clusters
fviz_cluster(heart_kmeans, 
             data = prepped_heart_df_cluster, 
             ellipse.type = "convex", 
             ggtheme = theme_classic(base_size = 20),
             ellipse.alpha = 0.1)

## what is the optimal number of clusters?
# uses silhouette method by default i.e., determines how well each object lies within its cluster (want to maximize)
fviz_nbclust(
  prepped_heart_df_cluster, 
  kmeans)

# can use `method = "wss"` to use the within-cluster sum of squares (want to minimize) 
# for identifying the optimal number of clusters
fviz_nbclust(
  prepped_heart_df_cluster, 
  kmeans,
  method = "wss")

# elbow method: identify the point where the rate of decrease in WCSS sharply changes
# the “elbow” point suggests the optimal number of clusters.

## run k-means clustering with 2 clusters
heart_kmeans_optimal = kmeans(prepped_heart_df_cluster, centers = 2, nstart = 25)
fviz_cluster(heart_kmeans_optimal, 
             data = prepped_heart_df_cluster, 
             ellipse.type = "convex", 
             ggtheme = theme_classic(base_size = 20),
             ellipse.alpha = 0.1)

# look at cluster characteristics in the updated clustering model - what can you tell about the sub-groups? 
aggregate(prepped_heart_df_cluster, by=list(cluster=heart_kmeans_optimal$cluster), mean)

# look at descriptive stats on the unscaled data so that it's easier to interpret
clustered_df <- prepped_heart_df %>% 
  # only keep the unscaled continuous variables
  select(c("age", "cigsPerDay", "totChol", "sysBP", "diaBP", "heartRate", "glucose")) %>% 
  # add a column containing the cluster assignments
  mutate(Cluster = heart_kmeans_optimal$cluster) %>%
  group_by(Cluster) %>%
  summarise_all("mean")
clustered_df

