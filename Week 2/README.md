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

3) Handle text and Categorical attributes:
	a. Ordinal encoder-*In Ordinal encoding, each unique category value is assigned an integer value. For example, "**red**" is **1*** , **"green"** is **2** , and **"blue"** is **3** .
```
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
```
b. One hot encoder
```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
```
The output is SciPy sparse matrix rather than NumPy array. This enables us to save space when we have a huge number of categories.
In case, we want to convert it to dense representation, we can do so with ***toarray()*** method.
4) Feature Scaling
	1) Min-Max scaling or normalization
		1) We subtract the minimum value of a feature from the current value and divide it by the difference between the minimum and the maximum value of that feature.
		2) Values are shifted and scaled so that they range between 0 and 1.
		3) Scikit-Learn provides *MinMaxScalar* transformer for this.
	2) Standardization
		1) We subtract mean value of each feature from the current value and divide it by the standard deviation so that the resulting feature has a unit variance.
		2) While normalization bounds values between 0 and 1, standardization does not bound values to a specific range.
		3) Standardization is less affected by the outliers compared to the normalization.
		4) Scikit-Learn provides StandardScalar transformation for feature standardization.
### Step 5 : Select and Train ML Model
1) Train and fit the model
```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(wine_features_tr, wine_labels)
```
2) Evaluate the model
```python
from sklearn.metrics import mean_squared_error
quality_predictions = lin_reg.predict(wine_features_tr)
mean_squared_error(wine_labels, quality_predictions)
```
3) Cross-Validation
```python
from sklearn.model_selection import cross_val_score
def display_scores(scores):
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())
scores = cross_val_score(lin_reg, wine_features_tr, wine_labels,
scoring="neg_mean_squared_error", cv=10)
lin_reg_mse_scores = -scores
display_scores(lin_reg_mse_scores)
```
4) Remedies for overfitting and underfitting 
	1) Overfitting
		1) More Data
		2) Simpler Model
		3) More constraints/Regularisation
	2) Underfitting
		1) Model with more capacity
		2) Less constraints/Regularization
### Step 6 : Fine Tune your Model
Usually there are a number of hyperparameters in the model, which are set manually. Tunning these hyperparameters lead to better accuracy of ML models.
1) GridSearchCV
		We need to specify a list of hyperparameters along with the range of values to try. It automatically evaluates all possible combinations of hyperparameter values using cross-validation.
```python
from sklearn.model_selection import GridSearchCV
param_grid = [

{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3,
4]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(wine_features_tr, wine_labels)
grid_search.best_params_
```
2) RandomizedSearchCV
		It selects a random value for each hyperparameter at the start of each iteration and repeats the process for the given number of random combinations.
```python
from sklearn.model_selection import RandomizedSearchCV
```
### Step 7 : Present your Solution
Before launch,
1) We need to present our solution that highlights learnings, assumptions and systems limitation.
2) Document  everything, create clear visualizations and present the model.
3) In case, the model does not work better than the experts, it may still be a good idea to launch it and free up bandwidths of human experts.
### Step 8 : Launch, Monitor, and Maintain your System
- Launch
	- Plug input sources
	- write test cases
- Monitor
	- System outages
	- Degradation of model performance
	- Sampling predictions for human evaluation
	- Regular assessment of data quality, which is critical for model performance
- Maintain
	- Train model regularly every fixed interval with fresh data.
	- Production roll out of the model.
## Introduction to Scikit-Learn
sklearn APIs are organized on the lines of our ML framework.

| *Scikit-learn*                                          | *ML Framework* |
| ------------------------------------------------------- | -------------- |
| Training data & preprocessing                           | Traning data   |
| Model subsumes loss function and optimization procedure | Model          |
| Model selection and evaluation                          | Loss Function  |
| Model inspection                                        | Optimization   |
### API Design Principles
sklearn APIs are well designed with the following principles:
- *Consistency* : All APIs share a simple and consistent interface.
- *Inspection* : The learnable parameters as well as hyperparameters of all estimator's are accessible directly via public instance variables.
- *Nonproliferation of classes* : Datasets are represented as NumPy arrays or Spicy sparse matrix instead of custom designed classes.
- *Composition* : Existing building blocks are reduced as much as possible.
- *Sensible Defaults* : Values are used for parameters that enables quick baseline building.

| *Transformers*                                                      | *Estimators*                                                         | *Predictors*                                                                   |
| ------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Transformers data.                                                  | Estimates model parameters based on training data & hyperparameters. | Makes prediction on dataset.                                                   |
| ==transform()== : for transforming dataset.                         | ==fit()==  method                                                    | ==predict()== : method that takes dataset as an input and returns predictions. |
| ==fit()== : learns parameters.                                      |                                                                      | ==score()== : method to measure quality of predictionsData Preprocessing       |
| ==fit_transform()== :fits parameters and *transform()* the dataset. |                                                                      |                                                                                |
![[Pasted image 20250215214902.png]]
#### Sklearn APIs 
##### Data API
Provides functionality for loading, generating and preprocessing the training data and test data.

| *Module*                     | *Funtionality*                                                  |
| ---------------------------- | --------------------------------------------------------------- |
| `sklearn.datasets`           | Loading datasets - custom as well as popular reference dataset. |
| `sklearn.preprocessing`      | Scaling, centering, normalization and binarization methods.     |
| `sklearn.feature_selection`  | Implements feature selection algorithms.                        |
| `sklearn.feature_extraction` | Implements feature extraction from raw data.                    |
##### Model API
Implements supervised and unsupervised models.

|                     *Regression*                     |    *Classification*    |
| :--------------------------------------------------: | :--------------------: |
| `sklearn.linear_model` (Linear, Ridge, Lasso models) | `sklearn.linear_model` |
|                   `sklearn.trees`                    |     `sklearn.svm`      |
|                                                      |    `sklearn.trees`     |
|                                                      |  `sklearn.neighbors`   |
|                                                      | `sklearn.naive_bayes`  |
|                                                      |  `sklearn.multiclass`  |

`sklearn.multipoint` implements multi-output classification and regression. 
`sklearn.cluster` implements many popular clustering algorithms.

##### Model Evaluation API
`sklearn.metrics` implements different APIs for model evaluation :
- Classification
- Regression
- Clustering
##### Model Selection API
`sklearn.model_selection` implements various model selection strategies like cross-validation, tuning hyper-parameters and plotting learning curves.
##### Model Inspection API
`sklearn.model_inspection` includes tools for model inspection.
## Data Loading
General dataset API has three main kind of interfaces :
- The dataset *loaders* are used to *load* toy datasets bundled with sklearn.
- The dataset *fetches* are used to *download and load* datasets from the internet.
- The dataset *generators* are used to *generate* controlled synthetic datasets.
### Dataset API
![[Pasted image 20250216114210.png]]
#### Dataset Loaders

| *Dataset Loader*     | * # samples (n) * | * # features (m) * | * # labels * | *Type*                      |
| -------------------- | ----------------- | ------------------ | ------------ | --------------------------- |
| `load_iris`          | 150               | 3                  | 1            | Classification              |
| `load_diabetes`      | 442               | 10                 | 1            | Regression                  |
| `load_digits`        | 1797              | 64                 | 1            | Classification              |
| `load_linnerud`      | 20                | 3                  | 3            | Regression (*multi-output*) |
| `load_wine`          | 178               | 13                 | 1            | Classification              |
| `load_breast_cancer` | 569               | 30                 | 1            | Classification              |
#### Dataset Fetchers

| *Dataset Loader*           | * # samples (n) * | * # features (m) * | * # labels * | *Type*                              |
| -------------------------- | ----------------- | ------------------ | ------------ | ----------------------------------- |
| `fetch_olivetti_faces`     | 400               | 4096               | 1 (40)       | Image Classification(*multi-class*) |
| `fetch_20newsgroups`       | 18846             | 1                  | 1 (20)       | Text Classification(*multi-class*)  |
| `fetc_lfw_people`          | 13233             | 5828               | 1 (5749)     | Image Classification(*multi-class*) |
| `fetch_covtype`            | 581012            | 54                 | 1 (7)        | Classification(*multi-class*)       |
| `fetch_rcvl`               | 804414            | 47236              | 1 (103)      | Classification(*multi-class*)       |
| `fetch_kddcup99`           | 4898431           | 41                 | 1            | Classification(*multi-class*)       |
| `fetch_california_housing` | 20640             | 8                  | 1            | Regression                          |
#### Dataset Generators
> Regression
- *make_regression()* produces regression targets as a spare random linear combination of random features with noise. The informative features are either uncorrelated or low rank.
> Classification
- Single-Label
	- *make_blobs()* and *make_classification()* create a bunch of normally-distributed clusters of points and then assign one or more clusters to each class thereby creating multi-class datasets.
- Multi-Label
	- *make_multilabel_classification()* generates random samples with multiple labels with a specific generative process and rejection sampling.
> Clustering
- *make_blobs()* generates a bunch of normally-distributed clusters of points with specific mean and standard deviations for each cluster.
### Loading External Libraries
- *fecth_openml()* fetches datasets from [openml.org]() , which is a public repository for ML data and experiments.
- *pandas.io* provides tools to read from common formats like CSV, excel, json, SQL
- *spicy.io* specializes in binary formats used in scientific computing like .mat and .arff .
- *numpy / routines.io* specializes in loading columnar data into NumPy arrays.
- *dataset.load_files*  loads directories to text files where directory name is a label and each file is a sample.
- *datasets.load_svmlight_files()* loads data in svmlight and libSVM sparse format.
- *skimage.io* provides tools to load images and videos in numpy arrays.
- *spicy.io.wavfile.read* specializes reading WAV file into numpy array.
## Data Transformation
### Types of transformers
sklearn provides a library of transformers for:
- Data cleaning (`sklearn.preprocessiong)
- Feature extraction (`sklearn.feature_extraction)
- Feature reduction
- Feature expansion(`sklearn.kernel_approximation)
### Transformer methods
- *fit()* method learns model parameters from a training set.
- *transform()* method applies the learnt transformation to the new data.
- *fit_transform()* performs function of both fit() and transform() methods and is more convenient and efficient to use.
Transformers are combined with one another or with other estimators such as classifiers or regressors to build composite estimators.

| *Tool*            | *Usuage*                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| Pipeline          | Chaining multiple estimators to execute a fixed sequence of steps in data preprocessing and modelling. |
| FeatureUnion      | Combines output from several transformer from them.                                                    |
| ColumnTransformer | Enables different transformations on different columns of data based on their types.                   |

---
