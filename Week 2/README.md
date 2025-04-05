## Steps in ML Project
### Step 1 : Look at the big picture
1) Frame the problem
	1) What are the input and output?
	2) What is the business objective?
	3) What is the current solution?
2) Select a performance measure
	1) Regression
		1) Mean Squared Error (MSE)
		2) Mean Absolute Error (MAE)
	2) Classification
		1) Precision
		2) Recall
		3) F1 Score
		4) Accuracy
	3) List and check the assumptions
### Step 2 : Get the Data
1) Check the data samples
2) Understand the significance of all features
3) Data Statistics
4) Create Test Set
	1) Avoid data snooping bias ( *a form of statistical bias manipulating data or analysis to artificially get statistically significant results* )
	2) Scikit learn provides a few functions to create test sets:
		1) Random Sampling-*randomly selects k% points in the test set*
		2) Stratified Sampling-*samples test examples in such a way that they are representative of the overall distribution, avoids bias that may arise due to random sampling*
### Step 3 : Data Visualization
1) Performed on the training data set
2) Standard correlation coefficient helps understand the relationship between features, visualize with heatmap
### Step 4 : Prepare Data for ML Algorithm
1) Separate features and labels from the labels from the training set.
2) Handling missing values and outliers.
	1) Sklearn SimpleImputer Class can impute missing values.
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

imputer.fit(wine_features)
tr_features = imputer.transform(wine_features)
```
