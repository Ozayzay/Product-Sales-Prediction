import sys
import pandas as pd
import numpy as np

# Kfold technique for cross validation to evaluate the model performance / Alternative to train_test_split
# 1 - Split the dataset into k equal partitions (or "folds")
# 2 - Use fold 1 as the testing set and the union of the other folds as the training set
# Helps to reduce over fitting by ensuring that every observation from the original dataset has the chance of appearing in training and testing set
from sklearn.model_selection import KFold
# For preprocessing 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# To handle imbalanced dataset by oversampling the minority class allows the model to get more training data on the minority class
# reducing the bias towards the majority class 
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def preprocess(data):

    df = data.copy()
    # Only use Historical data
    df = df[df['File_Type'] == 'Historical']
    # Drop unused columns
    df = df.drop(columns= ['Order' , 'SoldCount' , 'File_Type' ])

    # shuffle the data
    df = df.sample(frac = 1 , random_state = 1)

    y = df['SoldFlag']
    X = df.drop(columns = ['SoldFlag'])

    return X , y


def build_pipeline():

    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder( sparse_output = False , drop= "if_binary" ))
    ])

    # ColumnTransformer lets you apply different transformations to different columns in the dataset
    # Remainder = 'passthrough' will pass the columns that were not transformed because by default the columns that were not transformed are dropped
    preprocessor = ColumnTransformer ( transformers = [
        ('nominal' , nominal_transformer , ['MarketingType'])
    ], remainder = 'passthrough')


    model = Pipeline(steps=[
        ('preprocessor' , preprocessor),
        ('regressor' , RandomForestClassifier(random_state=1))
    ])

    return model



def main(in_directory):

    # Read the data

    data = pd.read_csv(in_directory)

    # Preprocessing

    X , y = preprocess(data)


    # Training / Validation 
    # KFold Cross Validation ( expects the data to be shuffled )

    accuracy_scores = []
    f1_scores = []

    kf = KFold(n_splits = 5)
    # kf.split(X) returns the indices of the data that should be used for training and testing in each iteration
    # Returns two arrays one for the training indices and one for the testing indices for each fold
    # For loop iterates over the 5 folds , in each iteration we use the indices to select the training and testing data
    for train_idx , test_idx in kf.split(X):
        X_train , X_test = X.iloc[train_idx, :] , X.iloc[test_idx, :]
        y_train , y_test = y.iloc[train_idx] , y.iloc[test_idx]

        # Address Class imbalance
        # So we are going to use RandomOverSampler - Which automatically brings the minority class up But before we have to bring the majority down to the average 
        # Find the number of samples we would need to bring the majority class down to the average
        num_samples = int( y_train.value_counts().mean() )
        # Find indices of the majority class
        majority_indices = y_train[y_train == 0].index

        # Fit the model for each fold
        model = build_pipeline()
        model.fit(X_train , y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        accuracy_scores.append( accuracy_score(y_test , y_pred) )
        # Pos = 1.0 because we are interested in the F1 score for the positive class - SoldFlag = 1
        # F1 score is better for imbalanced dataset ( only 17% of the data is SoldFlag = 1 )
        f1_scores.append (f1_score(y_test , y_pred , pos_label= 1.0))

    print("Accuracy: ", np.mean(accuracy_scores))
    print("F1: ", np.mean(f1_scores))




if __name__=='__main__':
    in_directory = sys.argv[1]
    # out_directory = sys.argv[2]
    main(in_directory)