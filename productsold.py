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

    y = df['SoldFlag']
    X = df.drop(columns = ['SoldFlag'])

    return X , y


def build_pipeline():

    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder( sparse = False , drop= "if_binary" ))
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
    data = pd.read_csv(in_directory)

    # print (data.head())

    # Preprocessing

    X , y = preprocess(data)

    # print (X.head())
    # print (y.head())

    ###### Pipe line ######

    model = build_pipeline()

    ###### Training / Validation ######
    # KFold Cross Validation




if __name__=='__main__':
    in_directory = sys.argv[1]
    # out_directory = sys.argv[2]
    main(in_directory)