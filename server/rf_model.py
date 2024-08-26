# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path



# Create the dataframe
def df_setup(data):
    df = pd.read_csv(data)
    return df

# Synthetic Minority Over-sampling Technique
def smote(X, y):
    smote_obj = SMOTE()
    X_resampled, y_resampled = smote_obj.fit_resample(X,y)
    return X_resampled, y_resampled

# Fit the model
def fit_model(model, X, y):
    model.fit(X, y)

# Perform a grid search for optimal hyperparameters
def grid_search(model, X, y):
    # Setup the parameter distribution
    param_dist = {
        'criterion':['gini','entropy'],
        'max_depth':[6, 7],
        'ccp_alpha':[0.03, .025, .02]
    }

    # Create and fit the grid
    grid = GridSearchCV(model, param_grid=param_dist, cv=10, n_jobs=-1)
    grid.fit(X, y)

    # *Should* return the randomforestclassifier that performed the best, already fitted
    return grid.best_estimator_ # returns instance of RandomForestClassifier() with the correct parameters that has been fitted
    
    # print(grid.best_params_)
    # # Determine the best hyperparameters
    # best_criterion = grid.best_params_.get('criterion')
    # best_max_depth = grid.best_params_.get('max_depth')
    # best_ccp_alpha = grid.best_params_.get('ccp_alpha')

    # param_dict = {'best_criterion':best_criterion, 'best_max_depth':best_max_depth, 'ccp_alpha':best_ccp_alpha}
    # return param_dict # Return dictionary containing the best hyperparameters

# Create and return the confusion matrix
def return_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)
    
# Create and return the classification report
def return_classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

# Serialize the model to a pickle file
def serialize_model(model, folderPath):
    Path(os.path.join(folderPath, 'model.pkl')).touch()
    with open(os.path.join(folderPath, 'model.pkl'), 'wb') as file:
        pickle.dump(model, file)

# Load the model from a pickle file
def deserialize_model(folderPath):
    with open(os.path.join(folderPath, 'model.pkl'), 'rb') as file:
        model = pickle.load(file)
        return model



# START TEST SECTION *******************************
def self_classification_decider(value, divider):
    if(value < divider):
        return 0
    return 1

# Create and return the confusion matrix
def return_cm_TEST(model, X_test, y_test, divider):
    y_pred = model.predict_proba(X_test)
    # Vector of probabilitys for placement in TCU
    arr = []
    for i in y_pred:
        arr.append(self_classification_decider(i[1], divider))

    return confusion_matrix(y_test, arr)

# Create and return the classification report
def return_cr_TEST(model, X_test, y_test, divider):
    y_pred = model.predict_proba(X_test)
    # Vector of probabilitys for placement in TCU
    arr = []
    for i in y_pred:
        arr.append(self_classification_decider(i[1], divider))

    return classification_report(y_test, arr)

# Create and return the classification report
def graph_results(model, X_test):
    y_pred = model.predict_proba(X_test)

    seaborn.histplot(y_pred.T[1]*100, cbar=True, bins=20)
    plt.show()
# END TEST SECTION *********************************



def prepPatientData(folderPath):
    """
    Use on patient data before inputting into getpatientTCUProba() as patient_data
    """
    df = pd.read_csv(os.path.join(folderPath, "single_patient_out.csv"), index_col=False)

    # Primarily for showcase, check if dataframe contains discharge disposition column, drop if it does
    if 'DISCH_DISP' in df.columns:
        df.drop(columns="DISCH_DISP", inplace=True)
    
    # Return dataframe
    return df
    # # Return ndarray of dataframe?
    # return df.to_numpy()

def getPatientTCUProba(model, patient_data):
    """
    model = random forest classifier
    patient_data = 2D array of shape: (1, featureCount) created from patient data we are trying to predict
    breakpoints = 2 item array, [bp1, bp2], where bp1 would be the breakpoint which below is considered Green, and bp2 is the breakpoint which above is considered red

    returns the probability according to our model that a patient will need to stay in a TCU.
    """
    return model.predict_proba(patient_data)[0][1]

def convertProbaToClass(probability, breakpoints):
    """
    probability = output from getPatientTCUProba(), probability of a patient being placed in a TCU
    breakpoints = 2 item array, [bp1, bp2], where bp1 would be the breakpoint which below is considered Green, and bp2 is the breakpoint which above is considered red

    returns the red/yellow/green classification (from breakpoints) of the probability inputted
    """
    if(probability < breakpoints[0]): return 'G'
    if(probability < breakpoints[1]): return 'Y'
    return 'R'

def getPerformanceMetrics(folderPath, divider):
    model = deserialize_model(folderPath=folderPath)
    
    rawDataFilePath = os.path.join(folderPath,"normalized_data.csv")
    dataframe = df_setup(rawDataFilePath)
    
    # Separate features and target variable
    
    X_set = dataframe.drop('DISCH_DISP', axis=1)
    y_set = dataframe['DISCH_DISP']

    X_train, X, y_train, y = train_test_split(X_set, y_set, test_size=0.3)

    print(return_cr_TEST(model, X, y, divider))
    print(return_cm_TEST(model, X, y, divider))
    graph_results(model, X)





# Main method
def FitModel(folderPath, createModel=True):
    # Set up the data. Might move this to another file later
    # Trains the model, returns after training.
    if not createModel:
        return deserialize_model(folderPath)
    
    rawDataFilePath = os.path.join(folderPath,"normalized_data.csv")
    dataframe = df_setup(rawDataFilePath)
    
    # Separate features and target variable
    X = dataframe.drop('DISCH_DISP', axis=1)
    y = dataframe['DISCH_DISP']

    X_resampled, y_resampled = smote(X, y)
    
    return grid_search(RandomForestClassifier().fit(X_resampled, y_resampled), X_resampled, y_resampled)

    # # Create the random forest model
    # rfc = RandomForestClassifier()

    # fit_model(rfc, X_resampled, y_resampled)

    # # UNCOMMENT IF YOU WANT TO PERFORM THE GRID SEARCH   
    # best_params_dict = grid_search(rfc, X_resampled, y_resampled)

    # rfc1 = RandomForestClassifier(criterion=best_params_dict.get('best_criterion'), max_depth=best_params_dict.get('best_max_depth'), ccp_alpha=best_params_dict.get('ccp_alpha'))
    # fit_model(rfc1, X_resampled, y_resampled)

    

    

    
    
