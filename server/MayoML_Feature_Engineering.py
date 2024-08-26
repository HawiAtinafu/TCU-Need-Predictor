import pandas
from numpy import nan, ndarray
import os

import sklearn.ensemble
# # For plotting/data visualization
# import seaborn
# import matplotlib.pyplot as plt

RAW_DATA_COLUMN_LIST = [
        'LOC_NAME','REGION','DISCHARGE_DEPT_NAME','PATIENT_ID','VISIT_ENCOUNTER_ID','HOSP_ADMSN_TIME','HOSP_DISCH_TIME',
        'DEPT_WHEN_ADMITTED','BMI','AGE_AT_ADMISSION','ADMIT_SOURCE','ED_CHIEF_COMPLAINT','ADMIT_DX_ID','ADMIT_DX_NAME',
        'ADMIT_DIAG_TEXT','FINAL_DX_NAME','FINAL_DX_ID','DISCH_DISP','ETHANOL_LEVEL_TEST','ETHANOL_LEVEL_RESULT_FLAG',
        'CARE_MGMT_CNSLT_EARLY_DISCHARGE','FINAL_LACE_SCORE','FINAL_CSSRS_SCORE','FINAL_AMPAC_SCORE',
        'COUNT_PAST_ED_VISITS','COUNT_PAST_HSP_VISITS']


def main(folderPath, CSVFileName, output_file_name, normalization_vals = None, OHE_keys = None, isDataSet = False):
    # Temporarily creating rawData variable to prevent definitions from getting mad, do not use until value is set via readcsv() function
    rawData = 0

    # Outdated, no longer being used/already served its purpose
    def FeatureEngineering(rawData, file_name=None, folder_path=None, engineer_features=False):
        """
        Notable variables in this function:\n
        FeatureCounts: 2D List of counts for each entry in each column of inputted dataframe.
        FeatureRawValues: 2D List of unique entries in each column of inputted dataframe. FeatureRawValues corresponds to FeatureCounts, so 'FeatureRawValue[i]: FeatureCounts[i]' will show an entry value followed by its count in the column that instance appeared.\n
        FeatureMap: Contains a list of all the unique entries combined with their original column name in the format 'column_name: entry'. For example, if column 'a' has two entries, '1' and '2', this list would look like ['a: 1', 'a: 2']\n
        FeatureListToKeyMap: Relating to the above list, this list contains the column_name that each entry would map to. Used as a method of access for the dataframe to avoid string manipulation to find column name for all created columns\n
        FeatureListToTokenMap: Identical to FeatureListToKeyMap, just using the unique entries found instead of the column names. Matches up with FeatureListToKeyMap, so dataframe[FeatureListToKeyMap[i]] will yield a column with the value FeatureListToTokenMap[i] in it. Each entry will only be addressed one time per column.\n
        *Untested if dataframe has multiple columns with identical names*
        """
        # Map out possible feature choices
        FeatureCounts = []
        FeatureRawValues = []
        FeatureRawValuesAndCounts = [] # Only used if printing to file
        for a in rawData.keys():
            tempDict = {}
            for b in rawData[a]: 
                if b not in tempDict:
                    tempDict[b] = 0
                tempDict[b] = tempDict[b]+1
            FeatureCounts.append(list(tempDict.values()))
            FeatureRawValues.append(list(tempDict.keys()))

            # For printing values to file ONLY
            if (file_name is not None):
                FeatureRawValuesAndCounts.append(str(a))
                for i in tempDict:
                    FeatureRawValuesAndCounts.append({str(i): str(tempDict[i])})
                FeatureRawValuesAndCounts.append("")           

        # Create the Feature Columns
        FeatureMap = [] # New Column Titles
        FeatureListToKeyMap = []
        FeatureListToTokenMap = []
        for i in range(len(FeatureRawValues)):
            for j in range(len(FeatureRawValues[i])):
                FeatureMap.append(f"{rawData.keys()[i]}: {FeatureRawValues[i][j]}") # Creating the list of column titles for the output dataframe
                FeatureListToKeyMap.append(rawData.keys()[i])
                FeatureListToTokenMap.append(FeatureRawValues[i][j])

        # Printing the txt file
        if (file_name is not None):
            writer = open(folder_path + file_name, 'w')
            for i in FeatureRawValuesAndCounts:
                writer.write(str(i))
                writer.write("\n")
            writer.close()

        if(engineer_features):
            rawData.drop('classification', axis='columns')
            Data = pandas.DataFrame(index=range(len(rawData[rawData.keys()[0]])), columns=FeatureMap, dtype=bool).replace(to_replace=True, value=False)
            Data = Data.astype(dtype=int, copy=True)

            for col in range(len(FeatureListToTokenMap)):
                for ind in range(len(rawData[rawData.columns[0]])):
                    if(rawData.at[ind,FeatureListToKeyMap[col]] == FeatureListToTokenMap[col]):
                        Data.at[ind,FeatureMap[col]] = 1

            return Data, FeatureMap, FeatureCounts, FeatureRawValues
    
    # Outdated, no longer being used/already served its purpose
    def combineColumns(data, columns_to_be_combined, output_column_name=None, separating_string="", data_type=str):
        """
        Combines two or more columns, with an optional string placed in between each value. Mostly for analysis, if used with numbers will combine rows by addition, just make sure to put separating_string as numerical value for 0\n
        data: dataframe containing all columns to be concatenated.\n
        columns_to_be_combined: list containing keys for all columns to be combined.\n
        output_column_name: name for final column created, if none provided, will default to list of columns_to_be_combined broken up by " : "\n
        separating_string: string to be placed inbetween each column when combined, will be cast to data_type before addition\n
        data_type: type of data to be used, untested for type besides string. 
        """
        if(columns_to_be_combined is None):
            columns_to_be_combined = data.columns
            
        if(output_column_name is None):
            output_column_name = str(columns_to_be_combined[0])
            for i in columns_to_be_combined[1:]:
                output_column_name = output_column_name + " : " + str(i)

        temp_indexes = [i for i in data.get(data.columns[0]).index]
        temp_df = pandas.DataFrame(data=data[columns_to_be_combined[0]], columns=[output_column_name], index=temp_indexes, dtype=data_type)
        temp_df[output_column_name].replace(to_replace=temp_df.iloc[0,0], value=data_type(), inplace=True)

        cns = [c for c in data.columns if c in columns_to_be_combined]

        for i in cns:
            if i == cns[0]:
                temp_df[output_column_name] = data[i].astype(data_type)
            else:
                temp_df[output_column_name] = temp_df[output_column_name].astype(data_type) + data_type(separating_string) + data[i].astype(data_type)
        return temp_df
    
    # Outdated, no longer being used/already served its purpose
    def traitCorrelationMapping(data, columns_to_correlate, new_column_name, bad_target_field=["Home or Self Care"]):
        """
        data: set of data containing all columns listed in columns_to_correlate\n
        columns_to_correlate: 2 item list of column names, column 2 is target column. First column is the column that will be counted.\n
        new_column_name: string, name of new column in DataFrame returned.\n
        bad_target_field: TO BE UPDATED. List, checks if the value at an index in columns_to_correlate is *not* in the list counting the total for each unique value in columns_to_correlate[0] to be used to find the correlation
        """
        TEMP_DICT = {}
        REPLACE_DICT = {}
        # Getting dictionaries of occurrence counts
        CORRELATION_DATAFRAME = pandas.DataFrame(data).get(columns_to_correlate).convert_dtypes(infer_objects=True)
        for i in range(len(CORRELATION_DATAFRAME[columns_to_correlate[0]])):
            if(CORRELATION_DATAFRAME[columns_to_correlate[0]][i] not in TEMP_DICT):
                # initializing each entry: [total occurrence count, number of times the occurrence follows Target parameters]
                TEMP_DICT[CORRELATION_DATAFRAME[columns_to_correlate[0]][i]] = [0, 0] 
            TEMP_DICT[CORRELATION_DATAFRAME[columns_to_correlate[0]][i]][0] += 1
            TEMP_DICT[CORRELATION_DATAFRAME[columns_to_correlate[0]][i]][1] += CORRELATION_DATAFRAME.iloc[i, 1]
        CORRELATION_DATAFRAME.drop(columns_to_correlate[1], inplace=True, axis='columns')

        for i in TEMP_DICT:
            REPLACE_DICT[i] = (TEMP_DICT[i][1] / TEMP_DICT[i][0])


        tmp_ls = []
        for i in CORRELATION_DATAFRAME[columns_to_correlate[0]]:
            tmp_ls.append(REPLACE_DICT[i])

        return pandas.DataFrame(data=tmp_ls, index=range(len(tmp_ls)), dtype=float, columns=[new_column_name])
    
    # Outdated
    def icd10CodeCategorization(rawData, code_column_name, output_column_modifier):

    
        """
        Breaks down columns of ICD-10 codes into the categories listed below variables\n\n
        rawData: Dataframe containing column\n
        code_column_name: Name of column containing ICD-10 codes\n
        output_column_modifier: String prepended to all output columns\n\n

        return: Dataframe containing one hot encoded columns for each of the columns listed. Note all column names will have output_column_modifier before them, so a positive A00-B99 with a column modifier "Admission" would have a 1 in 
        the column "Admission A00-B99" and 0 in other outputted columns. Note, After combining, all column names must be different so identical column modifiers between multiple calls will cause problems.\n\n
        
        A00-B99	Certain infections and parasitic diseases
        C00-D49	Neoplasms
        D50-D89	Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
        E00-E89	Endocrine, nutritional and metabolic diseases
        F01-F99	Mental, Behavioral and Neurodevelopmental disorders
        G00-G99	Diseases of the nervous system
        H00-H59	Diseases of the eye and adnexa
        H60-H95	Diseases of the ear and mastoid process
        I00-I99	Diseases of the circulatory system
        J00-J99	Diseases of the respiratory system
        K00-K95	Diseases of the digestive system
        L00-L99	Diseases of the skin and subcutaneous tissue
        M00-M99	Diseases of the musculoskeletal system and connective tissue
        N00-N99	Diseases of the genitourinary system
        O00-O9A	Pregnancy, childbirth, and puerperium
        P00-P96	Certain conditions originating in the perinatal period
        Q00-Q99	Congenital malformations, deformations and chromosomal abnormalities
        R00-R99	Symptoms, signs, and abnormal clinical laboratory findings, not elsewhere classified
        S00-T88	Injury, poisoning, and certain other consequences of external causes
        U00-U85 Codes for special purposes
        V00-Y99	External causes of morbidity
        Z00-Z99	Factors influencing health status and contact with health services
        """
        # Column names
        colList = ['A00-B99', 'C00-D49', 'D50-D89', 'E00-E89', 'F01-F99', 'G00-G99', 'H00-H59', 'H60-H95', 'I00-I99', 'J00-J99', 'K00-K95', 'L00-L99', 'M00-M99', 'N00-N99', 'O00-O9A', 'P00-P96', 'Q00-Q99', 'R00-R99', 'S00-T88', 'U00-U85', 'V00-Y99', 'Z00-Z99']
        # prepend modifier to col names
        for i in range(len(colList)):
            colList[i] = str(output_column_modifier) + " " + colList[i]

        # Create dataframe to be returned, fill with 0's
        df = pandas.DataFrame(index=rawData[code_column_name].index, columns=colList, dtype=int)
        df.fillna(value=0, inplace=True)

        # Break down using first 2 characters. First letter and first number characters are guaranteed to match and do not break format as the third character can. (ex. O9A.0)
        # If statement chain for placing in correct category
        for i in rawData[code_column_name].index:
            try:
                code = rawData[code_column_name][i]
                if(code.__class__ != str):
                    # Cannot perform string manipulation, ie. Null value, skip
                    pass
                elif(code[0] in 'AB'):
                    df[colList[0]][i] = 1
                elif(code[0] in 'C'):
                    df[colList[1]][i] = 1
                elif(code[0] in 'D'):
                    if(code[1] in '01234'):
                        df[colList[1]][i] = 1
                    elif(code[1] in '5678'):
                        df[colList[2]][i] = 1
                    else:
                        # InputError, invalid second character
                        print(code, ":  ", i)
                        pass
                elif(code[0] in 'E'):
                    df[colList[3]][i] = 1
                elif(code[0] in 'F'):
                    df[colList[4]][i] = 1
                elif(code[0] in 'G'):
                    df[colList[5]][i] = 1
                elif(code[0] in 'H'):
                    if(code[1] in '012345'):
                        df[colList[6]][i] = 1
                    elif(code[1] in '6789'):
                        df[colList[7]][i] = 1
                    else:
                        # InputError, invalid second character
                        print(code, ":  ", i)
                        pass
                elif(code[0] in 'I'):
                    df[colList[8]][i] = 1
                elif(code[0] in 'J'):
                    df[colList[9]][i] = 1
                elif(code[0] in 'K'):
                    df[colList[10]][i] = 1
                elif(code[0] in 'L'):
                    df[colList[11]][i] = 1
                elif(code[0] in 'M'):
                    df[colList[12]][i] = 1
                elif(code[0] in 'N'):
                    df[colList[13]][i] = 1
                elif(code[0] in 'O'):
                    df[colList[14]][i] = 1
                elif(code[0] in 'P'):
                    df[colList[15]][i] = 1
                elif(code[0] in 'Q'):
                    df[colList[16]][i] = 1
                elif(code[0] in 'R'):
                    df[colList[17]][i] = 1
                elif(code[0] in 'ST'):
                    df[colList[18]][i] = 1
                elif(code[0] in 'U'):
                    df[colList[19]][i] = 1
                elif(code[0] in 'VWXY'):
                    df[colList[20]][i] = 1
                elif(code[0] in 'Z'):
                    df[colList[21]][i] = 1
                else:
                    # InputError, didnt lead with capital letter
                    print(code, ":  ", i)
                    pass
            except TypeError:
                print(rawData[code_column_name][i], i)

        # return all encoded columns
        return df

    def simplifyTargetColumn(rawData, target_column_name='DISCH_DISP',
                            Non_TCU_placement_list= [
                            'Home or Self Care',
                            'Home-Health Care Svc',
                            'Home or Self Care w/ Planned Readmission',
                            'Hospice - Home'
                            ]
                            ):
        """
        Takes in information about the target column and simplifies column inplace, to a 0/1 where 1=TCU\n\n
        rawData: Dataframe containing target column.\n
        Non_TCU_placement_list: List containing all *non-TCU placement* results. If not in this tuple, will be considered TCU. Text must be exact.\n
        target_column_name: Name of target column in dataframe\n

        returns: dataframe containing column with simplified target
        """
        
        # Get target data
        Data = pandas.DataFrame(rawData.get(target_column_name))
        # Map function to convert data
        Data = Data.map(lambda x: int(not (x in Non_TCU_placement_list)))

        return Data

    def combineDataframes(dfList):
        """
        Combines a list of DataFrames into a single dataframe, then returns that dataframe.\n\n
        dflist: List of dataframes to combine. Will iterate through each column in each dataframe sequentially, so location will be determined by the order dataframes are placed in the list\n
        """
        df = pandas.DataFrame()
        for i in dfList:
            try:
                assert(pandas.DataFrame().__class__.__name__ == i.__class__.__name__)
                for j in i.keys():
                    df.insert(loc=len(df.keys()), column=j, value=i[j])
            except AssertionError:
                try:
                    assert(pandas.Series().__class__.__name__ == i.__class__.__name__)
                    df.insert(loc=len(df.keys()), column=i.name, value=i)
                except AssertionError:
                    # Not series or dataframe pass
                    pass
        return df
 
    def featureify(Data, input_column, target_column):
        """
        Generates a dict of all unique entries in the input column, and counts the number of times that column occurs compared to how many times it is positive for the target column
        Data: Dataframe containing both input and target column
        Input_column: column to get keys from
        target_column: column to count positives from
        """
        # Create entry dictionary
        feature_dict = {} # Dictionary containing 2 item lists for each unique entry in input_list: {unique_key: [# Times key appears in set, # Times key was placed in TCU]}
        for i in Data[input_column].index:
            temp = Data[input_column]
            try:
                feature_dict[Data.loc[i, input_column]][0] += 1
                feature_dict[Data.loc[i, input_column]][1] += Data.loc[i, target_column] # Verify if correct
            except KeyError:
                feature_dict[Data.loc[i, input_column]] = [1, Data.loc[i, target_column]]
        return feature_dict
    
    def Automated_Bucketing(rawData, input_column, max_buckets=5, target_column='DISCH_DISP'):
        """
        Buckets entries based on how similarly correlated they are to the output column

        return: 
        value_to_group: Dictionary containing all unique values in set as keys mapped to the bucket they are associated with as values
        group_set: Essentially inverse of value_to_group, Dict of each bucket number mapped to set object containing all unique inputs in that bucket
        """

        # Our working dataset
        Data = rawData.get([input_column, target_column])

        # One hot encode all unique entries if default input
        translation_dict = featureify(Data=Data, input_column=input_column, target_column=target_column)


        # labels. One will be strings for titles in the graph (original column names), the other to be used for dict keys 
        columns = list(translation_dict.keys())
        

        # Create dictionary. For each key in columns, will store the values that key is combined with. initialize with each key stores itself, adding any keys to the innermost dict as columns get combined
        combined_columns_dict = dict(translation_dict)
        for i in combined_columns_dict:
            combined_columns_dict[i] = [{i: 0}, translation_dict[i]]

        # Create array, to be filled and used in loop
        d = ndarray(shape=(len(columns),  len(columns)), dtype=float)


        # Boolean for repeat logic
        reduce_further = True
        # Var to store the number of buckets
        cur_bucket_count = -1
        # TODO add logic for stepsize and smart bucket numbers?
        
        # iterate if logic True
        while(reduce_further): 
            d = ndarray(shape=(len(columns),  len(columns)), dtype=float)
            # Set values in d to their correlation differences
            for m, i in enumerate(columns):
                for n, j in enumerate(columns):
                    # Fills ndArray d with values to put into dataframe
                    # Technically only need to update columns which were changed last round
                        if(m > n):
                            d[m][n] = abs(combined_columns_dict[i][1][1] / combined_columns_dict[i][1][0] - combined_columns_dict[j][1][1] / combined_columns_dict[j][1][0])

            # LocNums will be the index of the values which we want to update after iterating through the matrix
            LocNums = -1
            # LocVal is the value at LocNums, stored for testing purposes
            LocVal = 2
            IntLoc = (-1, -1)

            # Locates the index in d which has the highest correlation to another value, and stores that as a tuple in LocNums
            try:
                for m, i in enumerate(columns):
                    for n, j in enumerate(columns):
                        if m > n and d[m][n] < LocVal and (not combined_columns_dict[columns[m]][0] == combined_columns_dict[columns[n]][0]):
                            
                            LocNums = i
                            IntLoc = (m, n)
                            LocVal = d[m][n]
                            if(LocVal==0):
                                raise ImportError()
            except ImportError:
                # Used to break out of loop in the case that we detect a zero, no need to continue iterating
                pass


            # Goal for this area: Combine the data for the columns that are being combined in a few different areas
            # create dict of keys which need updating, convert to list after getting keys
            keys_to_update = {}

            # get initial unique keys from the location we want to combine
            # iterate through keys from combined columns, adding them to the keys to update
            def rec_key_addition(key):
                # Iterate through all keys associated with input key, if any arent in dict add and run again with new key
                for i in combined_columns_dict[key][0].keys():
                    try:
                        keys_to_update[i]
                    except KeyError:
                        keys_to_update[i] = 0
                        rec_key_addition(i)


            # add all locations with the same value to the list of keys to update, massive speed improvement until correlation difference is not 0.
            for a, i in enumerate(columns):
                if (combined_columns_dict[LocNums][1][1]/combined_columns_dict[LocNums][1][0]) == (combined_columns_dict[i][1][1]/combined_columns_dict[i][1][0]) or a in IntLoc:
                    keys_to_update[i] = 0
                    rec_key_addition(i)


            # Add new keys to all entries in combined_columns_dict
            for i in keys_to_update:
                for j in keys_to_update:
                    temp = combined_columns_dict[i][0]
                    try:
                        temp[j] = 0
                    except KeyError:
                        input("line 432 reached\n", i, j, temp)
                        combined_columns_dict[i][0] = {j: 0}

            # Should have all unique keys associated with current location
            keys_to_update = list(keys_to_update.keys())


            # combine values for all columns in list into value
            value = [0, 0]
            for k in keys_to_update:
                value[0] += translation_dict[k][0]
                value[1] += translation_dict[k][1]

            for k in keys_to_update:
                combined_columns_dict[k][1] = value


            for i in keys_to_update[1:]:
                try:
                    columns.remove(i)
                except ValueError:
                    # Key not in list, pass
                    pass
            # TEST
            cur_bucket_count = len(columns)
            # print(value, LocNums, LocVal, cur_bucket_count, "\n")
            if(cur_bucket_count<= max_buckets):
                reduce_further = False


            
        """
        # uncomment to graph data from function, **will pause program, DO NOT UNCOMMENT if process is automated**
        OHE_Data = pandas.DataFrame(data=d, index=columns, columns=columns, dtype=float)
        
        bucket_list = {}
        for i in combined_columns_dict.values():
            if not i[1] in bucket_list.values():
                bucket_list["Bucket "+str(len(bucket_list.keys())+1)+", Counts[agg,tcu+] "+str(i[1])+", correlation "+str(round(i[1][1]/i[1][0], 4)*100)+"%:\n"+str(list(i[0].keys()))+'\n']=i[1]
        print(OHE_Data)
        print("\nGroups Created:")
        for i in bucket_list.keys():
            print(i)
            

        seaborn.heatmap(OHE_Data)
        plt.xticks(ticks=range(len(OHE_Data.index)), labels=OHE_Data.index, minor=False, rotation=90)
        plt.yticks(ticks=range(len(OHE_Data.columns)), labels=OHE_Data.columns, minor=False)
        plt.show()
        """


        # Get unique groups
        # Group into buckets in the range of the max number of allowed buckets                              ...set objects exist :'(
        group_set = {}
        for a, i in enumerate(columns):
            group_set[a] = set(combined_columns_dict[i][0].keys())


        value_to_group = {}
        for i in group_set.keys():
            for j in group_set[i]:

                value_to_group[j] = i

        return value_to_group, group_set

    def one_hot_enc(value_to_group, group_set, rawData, input_column_name, output_columns_tag=None):
        """
        Takes a column and one hot encodes it, designed to be used in conjunction with Automated_Bucketing.
        
        return: Dataframe containing each index one hot encoded into the buckets inputted by value_to_group and group_set
        """
        # Set output col addition tag
        if(output_columns_tag is None):
            output_columns_tag = input_column_name+":"

        df = pandas.DataFrame(data=([0]*len(group_set.keys()) for _ in rawData.index), columns=group_set.keys(), index=rawData.index, dtype=float)
        for i in df.index:
            try:
                df.loc[i, value_to_group[rawData[input_column_name][i]]] = 1
            except KeyError:
                # Key not in value_to_group dict, meaning it is not a datapoint seen before.
                # Do nothing? (Should leave all as zeroes)
                pass
        
        # Renaming columns
        new_names = {}
        for i in df.columns:
            new_names[i] = output_columns_tag+str(i)
        df.rename(columns=new_names, inplace=True)
        return df
    
    def EthanolLevelTesttempfunc(x):
        # func used in converting ethanol test to integer, if possible to do via lambda, remove
        try:
            return int(x)
        except ValueError:
            return 0
        
    def MinMaxNormalizeCol(Data, column_name, min_max_dict):
        """
        Perform min/max normalization on a column with the provided mins/maxes on column inputted
        """
        Data[column_name] = (Data[column_name]-min_max_dict[column_name]['min']).apply(func= lambda x: max(x, 0)) / (min_max_dict[column_name]['max']-min_max_dict[column_name]['min'])



        

        

        

    # ***************************************************************************************************************************************
    

    rawDataFilePath = os.path.join(folderPath,CSVFileName)
    # read in all data from csv
    rawData = pandas.read_csv(filepath_or_buffer=rawDataFilePath, index_col=False, dtype=str)

    # TODO Add checks for if the file contains the correct columns






    # Drop useless/overlapping data            # Reason for removal
    rawData.drop(['ETHANOL_LEVEL_RESULT_FLAG', # Overlaps with much easier to manipulate numerical form of same data
                #   'ADMIT_DIAG_TEXT',    # Removed due to difficulty of parsing useful data from per-worker individualized notes
                'ED_CHIEF_COMPLAINT', # ADMIT_DX_ID is a more credible source of this information
                'ADMIT_DX_NAME', # Using ADMIT_DX_ID, same data
                'FINAL_DX_NAME', # Using FINAL_DX_ID, same data
                'FINAL_CSSRS_SCORE', # Advised against use
                
                'PATIENT_ID', # Identifier data, no correlation
                'VISIT_ENCOUNTER_ID', # Identifier data, no correlation

                # Removed
                'DISCHARGE_DEPT_NAME',
                'LOC_NAME',
                'ADMIT_SOURCE',
                'REGION',
                'DEPT_WHEN_ADMITTED',

                ],
                axis='columns',
                inplace=True)


    #***************************************          Replace acceptable Null values and start Simplifying Columns         ******************************************************
    # Turn target column into binary. Assign to raw data after for use with other functions
    SIMPLIFIED_TARGET_COLUMN = simplifyTargetColumn(rawData=rawData)
    rawData['DISCH_DISP'] = SIMPLIFIED_TARGET_COLUMN[SIMPLIFIED_TARGET_COLUMN.columns[0]]

    # Convert care management consult to integer. *Converting to string then using .astype() func to bypass FutureWarning
    CARE_MGMT_CNSLT_EARLY_DISCHARGE = rawData.get(['CARE_MGMT_CNSLT_EARLY_DISCHARGE']).replace(to_replace={'CARE_MGMT_CNSLT_EARLY_DISCHARGE': {'Care management consult for early screen discharge score': '1.0', nan: '0.0'}}).astype(dtype=float)
    rawData['CARE_MGMT_CNSLT_EARLY_DISCHARGE'] = CARE_MGMT_CNSLT_EARLY_DISCHARGE[CARE_MGMT_CNSLT_EARLY_DISCHARGE.columns[0]]

    # Replace missing values for each column in future NUMERICAL_VALUES
    numerical_columns = ['AGE_AT_ADMISSION','BMI','ETHANOL_LEVEL_TEST','FINAL_LACE_SCORE','FINAL_AMPAC_SCORE','COUNT_PAST_HSP_VISITS','COUNT_PAST_ED_VISITS']

    # Alcohol - Converting nans and unreadable data (ex. 'CANCELLED' and 'SEE COMMENT', also defaults '<10') to 0, otherwise take value as is
    rawData['ETHANOL_LEVEL_TEST'] = rawData['ETHANOL_LEVEL_TEST'].apply(EthanolLevelTesttempfunc)
    rawData['ETHANOL_LEVEL_TEST'].convert_dtypes(infer_objects=True)
    
    
    replace_dict = {
        'AGE_AT_ADMISSION': {nan: 38}, # AGE - Substitute overall average 38 (census.gov)
        'BMI': {nan: 26.65}, # BMI - Substitute overall average 26.55 (cdc.gov)
        'COUNT_PAST_ED_VISITS': {nan: 0}, # Past ED visits, Past HSP visits - 0, likely to mean search did not come up with results
        'COUNT_PAST_HSP_VISITS': {nan: 0},
        'FINAL_AMPAC_SCORE': {nan: 24}, # AMPAC Score - Assumed that NULL means test not needed, most likely normal - 24
        'FINAL_LACE_SCORE': {nan: 78} # LACE Score - 78 for median during testing, unknown what actual score is, change.
    }


    if(isDataSet):
        for i in replace_dict.keys():
            rawData.replace(to_replace=replace_dict, inplace=True)

    # take out useful numerical data.
    NUMERICAL_VALUES = rawData.get(numerical_columns)
    # Pray they actually got converted to float
    NUMERICAL_VALUES = pandas.DataFrame(NUMERICAL_VALUES, dtype=float)

    # Columnwise mean normalization on NUMERICAL_VALUES

    for i in numerical_columns:
        if(isDataSet):
            # Store values for use with individual patients
            normalization_vals[i] = {'min': NUMERICAL_VALUES[i].min(), 'max': NUMERICAL_VALUES[i].max()}
        MinMaxNormalizeCol(NUMERICAL_VALUES, i, normalization_vals)


    #***************************************          Remove *unacceptable* Null values         ******************************************************

    """             TODO - ON HOLD
    What defines an 'unacceptable' null value?
    1. A value that cannot be realistically substituted
        ex. admission source, admit/disch time, etc.
    2. 


    Per Patient/Incident:
    remove overlapping ENCOUNTER_ID
    remove no placement, null DISCH_DISP


    remove too many/too important null values?
        ex. if all diagnosis are missing (ADMIT_DX_ID,ADMIT_DX_NAME,ADMIT_DIAG_TEXT,FINAL_DX_NAME,FINAL_DX_ID,) possibly in conjunction with other data (BMI, AGE, *[test scores])
        ex. count important ^ columns, if >X number of columns are null or not enough diagnosis information can be gleaned 

    After, reindex as otherwise the entire program will have to be double checked to verify that no functions fail (ex. for i in range(len(df.index)):  instead of  for i in df.index:)
    """



    #***************************************          Create new features         ******************************************************
    # # Duration of stay feature, TODO Keep? May provide misleading results 
    # # If used, will need to update to save the data in normalization_vals
    # DURATION_OF_STAY = pandas.DataFrame(data=rawData.get(['HOSP_DISCH_TIME','HOSP_ADMSN_TIME']))

    # DURATION_OF_STAY['HOSP_ADMSN_TIME'] = pandas.to_datetime(DURATION_OF_STAY['HOSP_ADMSN_TIME'])
    # DURATION_OF_STAY['HOSP_DISCH_TIME'] = pandas.to_datetime(DURATION_OF_STAY['HOSP_DISCH_TIME'])
    # # Calculate Days Stayed
    # DURATION_OF_STAY['DURATION_STAYED'] = (DURATION_OF_STAY['HOSP_DISCH_TIME']-DURATION_OF_STAY['HOSP_ADMSN_TIME']).dt.days
    # DURATION_OF_STAY.drop(['HOSP_ADMSN_TIME', 'HOSP_DISCH_TIME'], inplace=True, axis='columns')
    # # Columnwise mean normalization on stay duration
    # DURATION_OF_STAY = (DURATION_OF_STAY-DURATION_OF_STAY.min())/(DURATION_OF_STAY.max()-DURATION_OF_STAY.min())



    # Break down idc-10 codes into columns
    # Most Likely Cause:
    # If final DX exists, use it otherwise use admit DX
    tempDF = rawData.get(['FINAL_DX_ID']).rename(columns={'FINAL_DX_ID':'ASSUMED_DX_ID'})
    test_mask_replace_val = {}
    for i in tempDF.index:
        if tempDF['ASSUMED_DX_ID'][i] is nan:
            test_mask_replace_val[i] = rawData['ADMIT_DX_ID'][i]
        else:
            test_mask_replace_val[i] = tempDF['ASSUMED_DX_ID'][i]
            
    # Inserting 'ASSUMED_DX_ID' into rawData at last slot
    rawData.insert(loc=len(rawData.columns), column='ASSUMED_DX_ID', value=pandas.Series(data=test_mask_replace_val, dtype=rawData['FINAL_DX_ID'].dtype, name='ASSUMED_DX_ID'))

    #Cut off all but 5 chars of dx id (A01.2|...) Might be worth testing with only 3 characters (A01|...)
    for i in rawData.index:
        temp = rawData.loc[i, 'ASSUMED_DX_ID']
        try:
            rawData.loc[i, 'ASSUMED_DX_ID'] = temp[:min(len(temp), 5)]
        except TypeError:
            # Type is not indexable (nan), pass
            pass


    # Add columns to be one hot encoded into 'one_hot_enc_input_list'
    # Output will be a list of encoded dataframes in 'one_hot_enc_output_list'

    one_hot_enc_input_list = [
                                # ['AGE_AT_ADMISSION', 9], For testing
                                ['ASSUMED_DX_ID', 20]
                            ]
    one_hot_enc_output_list = []

    for i in one_hot_enc_input_list:
        if(isDataSet):
            value_to_group, group_set = Automated_Bucketing(rawData=rawData, input_column=i[0], max_buckets=i[1])
            OHE_keys[i[0]] = {'value_to_group': value_to_group,
                              'group_set': group_set
                              }

        one_hot_enc_output_list.append(one_hot_enc(OHE_keys[i[0]]['value_to_group'], OHE_keys[i[0]]['group_set'], rawData, i[0]))




    #***************************************          Final combination           ******************************************************    
    # All data combined into one dataframe
    COMBINED_DATAFRAME = combineDataframes([ 
                                            CARE_MGMT_CNSLT_EARLY_DISCHARGE, 
                                            NUMERICAL_VALUES, 
                                            # DURATION_OF_STAY,  
                                            *one_hot_enc_output_list, # Need to empty out the list
                                            SIMPLIFIED_TARGET_COLUMN
                                            ])

    # Print the file to its designated position, overwriting original file if it exists
    output_path = os.path.join(folderPath,output_file_name)
    COMBINED_DATAFRAME.to_csv(output_path, index_label=False, index=False)






# For the demo only, useless otherwise.
def SHOWCASEONLY_findPatient(folderPath, CSVFileName, encounterID):
    rawDataFilePath = os.path.join(folderPath,CSVFileName)
    rawData = pandas.read_csv(filepath_or_buffer=rawDataFilePath, index_col=False, dtype=str)

    encounter = rawData[rawData['VISIT_ENCOUNTER_ID'] == encounterID]

    # Cause error if no index attribute (failed) or return if no valid encounter
    if len(encounter.index) == 0:
        return None
    
    # if multiple, remove all past the first instance found in dataframe
    if len(encounter.index) != 1:
        encounter.drop(index=encounter.index[1:], inplace=True, errors='ignore')

    return pandas.DataFrame(encounter, columns=RAW_DATA_COLUMN_LIST)

    


