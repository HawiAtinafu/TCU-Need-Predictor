import Data_Cleaner
import rf_model
import os
import traceback
from shutil import copy as copyFile
from pathlib import Path


class Predictor():

    # TODO Put in safety checks EVERYWHERE, crashes here mean whole application can go kaput or corrupt

    def __init__(self, folderPath, insertData='', createModel=True, createCleaner=True) -> None:
        """
        Call constructor whenever you want to retrain the model, update the dataset, change a file, or if an unfixable error comes up\n\n
        folderPath: string for the directory of the folder to place storage data in. Files in this folder will have constant names. Created if doesnt exist.\n
        insertData: String. If not blank (''), must be string or Path object for file path to a set of data as a csv. Will overwrite current set if it exists\n
        createModel: boolean, if False will attempt to read model from pkl file instead of training, must be True if data folder is empty\n
        createCleaner: boolean, if False will attempt to read data for cleaner from pkl file, must be True if data folder is empty
        """
        # Make cleaner and do initial clean of dataset
        try:
            # Guaranteed to fail?
            Path(folderPath).mkdir()
        except FileNotFoundError:
            # Folder or one or more parents does not exist. Creating file
            print("Path not found, creating...")
            if(insertData == ''):
                raise PredictorError(Exception(), code=6) # if creating a folder, must insert data as well
            Path(folderPath).mkdir(parents=True)
        except FileExistsError:
            # Folder already exists, will use this folder
            pass
        
        # Save folder path for later use
        self.folderpath = folderPath

        
        if(insertData != ''):
            try:
                self.storeDataIn(insertData, 'Set')
            except PermissionError as e:
                raise PredictorError(e, 2)
            except FileNotFoundError as e:
                raise PredictorError(e, 1)
            

        # prep data for initial model creation and store cleaner pkl files, or validate cleaner pkl files
        try:
            self.cleaner = Data_Cleaner.DataCleaner(folderPath=folderPath,
                                                    CleanSet=createCleaner)
        except FileNotFoundError as e:
            # TODO if failed, return ____
            raise PredictorError(e, code=1)
        
        # train and store model or validate model pkl file exists
        try:
            model = rf_model.FitModel(self.folderpath, 
                                      createModel=createModel)

            if(createModel):
                rf_model.serialize_model(model=model, folderPath=folderPath)
            print("Model Generation: Success")
        except FileNotFoundError as e:
            # TODO if failed, return ____
            raise PredictorError(e, code=1)


    def predictPatient(self, clearDataAfter=False, colorScale=False, breakpoints=None):
        """ 
        Predicts the probability for a patient to be placed in TCU\n
        Must have valid patient_data_in.csv present in folder before using\n
        to get a R/Y/G indication instead of a decimal, set colorScale=True and input the breakpoints
        in the format [a, b] where Green < a, Red > b, and a <= Yellow <= b. [Green |a| Yellow |b| Red]
        Note, the distribution is not perfect. There will likely never be a probability > .90 or < .10, 
        values must be adjusted accordingly. To test this, see model rf_model, (ctrl+f "TEST SECTION***").
        """
        # Clean the data for the individual
        try:
            self.cleaner.CleanIndividualData()
        except AttributeError as e:
            # Something went wrong and the cleaner was never confirmed to be working
            raise PredictorError(e, code=-1)
        
        # read in the model for use
        model = -1
        try:
            model = rf_model.deserialize_model(folderPath=self.folderpath)
        except FileNotFoundError as e:
            raise PredictorError(e, code=1)
        except PermissionError as e:
            raise PredictorError(e, code=2)
            

        # Setting temp variable, once bugtested and safetychecked, combine into one line
        returnVal = rf_model.getPatientTCUProba(model, rf_model.prepPatientData(self.folderpath))

        if(clearDataAfter):
            try:
                self.clearData(patientNormalized=True, patientRaw=True)
            except PermissionError as e:
                raise PredictorError(e, code=2)

        if(colorScale):
            if breakpoints == None:
                raise PredictorError(code=5)
            return rf_model.convertProbaToClass(probability=returnVal, breakpoints=breakpoints)
        return returnVal
    
    def storeDataIn(self, csvOriginalFilePath, dataType):
        """
        dataType must be either 'Patient' or 'Set', if neither will return error.
        copies file to data folder, overwrites if file already exists.
        """
        if(dataType=='Patient'):
            Path(os.path.join(self.folderpath, 'single_patient_in.csv')).touch()
            copyFile(csvOriginalFilePath, os.path.join(self.folderpath, 'single_patient_in.csv'))

        elif(dataType=='Set'):
            Path(os.path.join(self.folderpath, 'raw_dataset.csv')).touch()
            copyFile(csvOriginalFilePath, os.path.join(self.folderpath, 'raw_dataset.csv'))

        else:
            raise PredictorError(code=4)
        
    def SHOWCASEONLY_findPatient(self, patientID):
        """
        Searches the raw dataset for a visitorID and if found, stores that visitors data as the next patient to be tested.
        This function is to create data for demonstration purposes only. Assumes there is a dataset already existing in the file location.\n

        returns: Boolean for if the change was successful
        """
        try:
            return self.cleaner.SHOWCASEONLY_findPatient(patientID)
        except AttributeError as e:
            # Cleaner not initialized or 
            raise PredictorError(e, code=3)
    
    def clearData(self, setRaw=False, setNormalized=False, patientRaw=False, patientNormalized=False, cleanerData=False, modelData=False):
        """
        Clears the data from the set, specify what by changing the default paramenters. \n\n
        Notes:\n
        modelData=True and cleanerData=True will require a retrain of the model before predicting.\n
        setRaw=True will require a full recleaning and retraining for accurate results on next use, better to
        just use constructor with same filepath and set insertData='' to the correct path.\n
        setNormalized=True will raise errors if trying to retrain model without cleaning first\n
        patientNormalized=True will not affect model or prediction

        """
        def __checkandremove__(p):
            if(os.path.exists(p)):
                try:
                    os.remove(p)
                except PermissionError as e:
                    raise PredictorError(e, code=2)
            
        
        if(setRaw):
            p = os.path.join(self.folderpath, 'raw_dataset.csv')
            __checkandremove__(p)
        if(setNormalized):
            p = os.path.join(self.folderpath, 'normalized_data.csv')
            __checkandremove__(p)
        if(patientRaw):
            p = os.path.join(self.folderpath, 'single_patient_in.csv')
            __checkandremove__(p)
        if(patientNormalized):
            p = os.path.join(self.folderpath, 'single_patient_out.csv')
            __checkandremove__(p)
        if(cleanerData):
            p = os.path.join(self.folderpath, 'normalized_data.pkl')
            __checkandremove__(p)
            p = os.path.join(self.folderpath, 'ohe_keys.pkl')
            __checkandremove__(p)
        if(modelData):
            p = os.path.join(self.folderpath, 'model.pkl')
            __checkandremove__(p)
           



# Adding type of error for Application to catch
class PredictorError(Exception):
    codeDict = {
                -1: 'Error code not implemented',
                1: 'File doesnt exist',
                2: 'Permission to write to file not granted',
                3: 'Must initialize cleaner and/or model first',
                4: 'Invalid input for a function',
                5: 'Breakpoints not set',
                6: 'If creating a folder, must add data file as well.'
                }
     
    def translateMessage(self):
        try:
            return PredictorError.codeDict[self.code]
        except KeyError:
            return 'Error code not found'


    def __init__(self, *args: object, code=None) -> None:
        super().__init__(*args)
        self.code = code
        self.message = self.translateMessage()

    def printStackTrace(self):
        traceback.print_exc()



# TESTING, REMOVE TODO
if __name__ == '__main__':
    def printHelp():
        print(
                    "'test' to test patient",
                    "'train' to retrain the model",
                    "'performance' to print model accuracy",
                    "'getdata' to retrieve a guinea pig from the dataset",
                    "'cleardata' to empty file",
                    "'t' arbitrary function to test, variable",
                    sep = "\n"
                )
    # *********************************************** ENTER YOUR FILE PATH HERE ***********************************
    # ex: "C:\\Users\\nrsch\\OneDrive\\Desktop\\test", should be folder which will contain dataset
    filepath = "C:\\Users\\hawia\\MayoTest"


    command = input("Enter action to perform (Usually start with 'train'):  ")
    # Creating with personal file path for testing only
    obj = None
    while(command != "exit" and command != "q" and command != "quit"):
        try:
            if command == "help":
                printHelp()
            
            elif command == 'test':
                # Test an individual patient
                # TODO add testing for errors
                print("TCU Prediction:   " + str(obj.predictPatient()), "Classification:   " + str(obj.predictPatient(colorScale=True, breakpoints=(.3, .7))), sep="\n")


            elif command == 'train':
                # Train the model
                if(input("Read model/cleaner from pkl file?   ") == 'n'):
                    obj = Predictor(filepath)
                else:
                    obj = Predictor(filepath, createModel=False, createCleaner=False)

            elif command == 'performance':
                rf_model.getPerformanceMetrics(folderPath=obj.folderpath, divider=float(input("Enter decimal for divider:   ")))

            elif command == 'getdata':
                encounterID = input("Enter the encounterID id (ex. '3017085'):   ")
                result = obj.SHOWCASEONLY_findPatient(encounterID)
                if(result):
                    print("Success!!!")
                else:
                    print("Encounter not found")
            
            elif command == 'cleardata':
                print("for each, enter 'y' to remove")
                obj.clearData(
                    setRaw=input('remove raw set data?   ')=='y',
                    setNormalized=input('remove normalized set data?   ')=='y',
                    patientRaw=input('remove raw patient data?   ')=='y',
                    patientNormalized=input('remove normalized patient data?   ')=='y',
                    cleanerData=input('remove cleaner?   ')=='y',
                    modelData=input('remove model?   ')=='y',
                )

            elif command == 't':
                # Arbitrary test function to test set code
                pass

            else:
                print("command not found, enter 'help' for a list of commands.")
        except Exception:
            traceback.print_exc()

        command = input('Enter action to perform:  ')
        
    print("Program Exiting")
        


        