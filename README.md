Product Sales Prediction

This project aims to predict whether a product will sell in the next six months based on historical sales data. By leveraging machine learning techniques, we analyze key product attributes to determine sales likelihood, providing actionable insights for inventory optimization.

Project Overview

This project utilizes a dataset of historical sales and active inventory records to build a machine learning model for predicting product sales (SoldFlag = 1). The goal is to assist in inventory decision-making by identifying products likely to sell versus those that should be removed.


During Testings: 
1. 
When we only hot encoded the Marketing type we got a pretty bad F1 score and accuracy score of 83 % - which can be misleading because of the fact that we had 17% of the products were sold 
the accuracy comes from only predicting the majority class ( Soldfalg = 0 )
This is re-enforced by how the F1 score is so low i.e 24% basically indicating that we are not doing a well job at predicting the Positive class likely due to class imbalances 

2. 
Try Class Re-balancing 

For SoldFlag = 0 we have 63 000 data points  
For Sold Flag = 1.0 we have 12 996 data points

We are going to use both Undersampling- Decrease Majority class and OverSampling - Increase  minority class 

With this version we have :
Accuracy:  0.7663166673015191
F1:  0.4362355241494528

Which is a lot better 

3. Try putting the ReleaseNumber - column as a categorical feature as well - So basically one hot encode it as well 

Because we are thinking about taking it as a non-ordinal groupings eg different product lines ( As the Kaggle data set didn't provide any information about this column)
When we weren't doing this we thought of it as something Ordinal - and treated it as numeric. This basically meant if the release number 

Represented a progression eg newer releases means higher number - it could relate to newer - technology , better performance and hence more sold 
Why Does This Matter in a Random Forest Model?

Random Forests and Categorical Data:
	•	Random forests can handle numeric features well, but numeric encoding can misrepresent categorical data.
	•	For example, treating ReleaseNumber as numeric might lead the model to assume that ReleaseNumber=10 is twice as impactful as ReleaseNumber=5, which may not make sense.
Benefit of One-Hot Encoding:
	•	By one-hot encoding, you allow the model to consider each ReleaseNumber as an independent feature.
	•	This removes any artificial relationships imposed by the numeric representation, making it easier for the model to focus on the actual patterns in the data.

Trade-Offs of One-Hot Encoding

Advantages:
	•	Captures non-linear relationships between categories.
	•	Eliminates incorrect assumptions about ordinal relationships.
Disadvantages:
	•	Increases dimensionality (number of features), especially if there are many unique ReleaseNumber values.
	•	May require additional memory and computation.


Accuracy:  0.7713564128180178
F1:  0.4370438839808893

- Result for doing Release Number as one hot encoded also increased the accuracy 
-------

Dataset Description

The dataset includes both historical sales data and active inventory. For this project:
	•	Only historical data is used, as active inventory lacks the target column SoldFlag.
	•	The dataset includes:
	•	Product identifiers and attributes.
	•	Sales data for the past 6 months.
	•	A binary target variable (SoldFlag) indicating whether a product sold during that period.

Key Columns:
	•	SKU_number: Unique product identifier.
	•	File_Type: Indicates whether the record is historical or active.
	•	SoldFlag: Target variable (1 = sold, 0 = not sold).
	•	SoldCount: Number of items sold (removed to avoid data leakage).
	•	MarketingType: Categorical variable for product marketing strategy.
	•	ReleaseNumber: Release version of the product.
	•	StrengthFactor and PriceReg: Numerical attributes representing product characteristics.


Key Objectives

	1.	Predictive Modeling:
	•	Build a classification model to predict SoldFlag.
	2.	Class Imbalance Handling:
	•	Address the imbalance (only ~17% products sold) using techniques like oversampling and undersampling.
	3.	Performance Evaluation:
	•	Use metrics like F1 Score and Accuracy to assess model performance.
	4.	Insights for Inventory Optimization:
	•	Provide probability scores for each product to aid decision-making.


Techniques and Tools

Tools

	•	Languages: Python
	•	Libraries:
	•	Data Manipulation: Pandas, NumPy
	•	Data Preprocessing: Scikit-learn, Imbalanced-learn
	•	Modeling: Scikit-learn (Random Forest Classifier)
	•	Evaluation: Cross-validation, Accuracy, F1 Score

Techniques

	1.	Data Preprocessing:
	•	Dropped irrelevant columns (Order, SKU_number, SoldCount).
	•	Handled categorical variables using binary encoding and one-hot encoding.
	•	Addressed missing values by filtering historical records.
	2.	Class Imbalance Handling:
	•	Combined Random Oversampling and Undersampling to balance the dataset.
	3.	Modeling:
	•	Used a Random Forest Classifier for robust predictions.
	•	Built a pipeline integrating preprocessing and modeling steps.
	4.	Evaluation:
	•	Performed K-Fold Cross-Validation to assess model performance.
	•	Focused on F1 Score for imbalanced classification.