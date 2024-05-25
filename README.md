#We'll apply knn classifier on MNIST data using PCA.
We did following analysis
1. We first picked top 3 PCs to visualize the data in 3D space.
2. We then ran a PCA with number of principal components equal to the original feature size. After that, we identified 
the number (called threshold) of principal components that are capturing 90% variance of the original data. We observed
that we could transpose the 770+ original features to 220 principal components to capture 90% variance. 
3. We ran the k neighbor classifier on the PCA (with 220 components) transformed data and achieved maximum accuracy.
