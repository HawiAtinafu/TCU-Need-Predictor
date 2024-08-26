import MayoML_Feature_Engineering as fe
import pandas as pd
from numpy import nan, ndarray
import os
import pickle
import ML_Predictor


class DataCleaner():
    def __init__(self, folderPath, CleanSet=True) -> None:
        # Stores args for Data normalization
        self.folderPath = folderPath
        self.DATASET_CSVFileName = "raw_dataset.csv"
        self.DATASET_output_file_name = "normalized_data.csv"

        self.PATIENT_CSVFileName = "single_patient_in.csv"
        self.PATIENT_output_file_name = "single_patient_out.csv"

        normalization_vals = dict() 
        OHE_keys = dict()

        self.isDataSet = CleanSet

        if(CleanSet):
            try:
                self.CleanSetData(OHE_keys, normalization_vals)
            
                self.store_normalization_vals(normalization_vals)
                self.store_OHE_keys(OHE_keys)
                self.isPrepped = True
            except FileNotFoundError as e:
                # Folder location invalid or does not contain raw_dataset.csv
                raise ML_Predictor.PredictorError(e, 1)
        else:
            try:
                self.retrieve_normalization_vals()
                self.retrieve_OHE_keys()
                self.isPrepped = True
            except FileNotFoundError as e:
                # Couldnt find one or more of the pickle files, need to clean data
                raise ML_Predictor.PredictorError(e, 1)
        
        #Testing, remove TODO
        print("Data Cleaning: Success")
        
        

        

    def SHOWCASEONLY_findPatient(self, encounterID):
        x = None
        try:
            x = fe.SHOWCASEONLY_findPatient(self.folderPath, self.getDataSetInputFileName(), encounterID)
        except AttributeError as e:
            raise ML_Predictor.PredictorError(e, code=3)

        if x.__class__==None.__class__ or len(x.index)==0:
            return False
        else:
            x.to_csv(os.path.join(self.folderPath, self.getPatientInputFileName()), index_label=False, index=False)
            return True

    def CleanIndividualData(self):
        normalization_vals = self.getNormalizationValues()
        OHE_keys = self.getOHEKeys()
            
        rv = -1
        try:
            rv = fe.main(folderPath=self.folderPath, 
                       CSVFileName=self.PATIENT_CSVFileName, 
                       output_file_name=self.PATIENT_output_file_name,
                       normalization_vals=normalization_vals,
                       OHE_keys=OHE_keys,
                       isDataSet=False
                       )
        except FileNotFoundError as e:
            raise ML_Predictor.PredictorError(e, code=1)

    def CleanSetData(self, OHE_keys, normalization_vals):
        fe.main(folderPath=self.folderPath, 
                CSVFileName=self.DATASET_CSVFileName, 
                output_file_name=self.DATASET_output_file_name,
                normalization_vals=normalization_vals,
                OHE_keys=OHE_keys,
                isDataSet=True
                )

    # Serialize data to a pickle file
    def serialize_data(self, obj, folderPath, filename):
        with open(os.path.join(folderPath, filename), 'wb') as file:
            pickle.dump(obj, file)

    # Load data from a pickle file
    def deserialize_data(self, folderPath, filename):
        with open(os.path.join(folderPath, filename), 'rb') as file:
            model = pickle.load(file)
            return model
        
    def store_OHE_keys(self, OHE_keys):
        self.serialize_data(OHE_keys, folderPath=self.folderPath, filename='ohe_keys.pkl')

    def store_normalization_vals(self, normalization_vals):
        self.serialize_data(normalization_vals, folderPath=self.folderPath, filename='normalized_data.pkl')

    def retrieve_OHE_keys(self):
        return self.deserialize_data(folderPath=self.folderPath, filename='ohe_keys.pkl')
    
    def retrieve_normalization_vals(self):
        return self.deserialize_data(folderPath=self.folderPath, filename='normalized_data.pkl')
    
    def getDataSetInputFileName(self):
        return self.DATASET_CSVFileName

    def getPatientInputFileName(self):
        return self.PATIENT_CSVFileName

    def getDataSetOutputFileName(self):
        return self.DATASET_output_file_name

    def getPatientOutputFileName(self):
        return self.PATIENT_output_file_name

    def getFolderPath(self):
        return self.folderPath

    def isDataPrepared(self):
        return self.isPrepped
    
    def getNormalizationValues(self):
        return self.retrieve_normalization_vals()
    
    def getOHEKeys(self):
        return self.retrieve_OHE_keys()