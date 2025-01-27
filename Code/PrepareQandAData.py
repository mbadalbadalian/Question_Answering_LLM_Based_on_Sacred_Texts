import pandas as pd

#################################################################################################################
### Other Functions

#LoadDF Function
def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

#CreatePreparedQAndAData Function
def CreatePreparedQAndAData(Q_and_A_DF,ESV_Bible_DF,ESV_Bible_Book_id_DF,book_mapping_csv_filepath,prepared_Q_and_A_DF_filepath):
    #Creating the Q&A dataset
    prepared_Q_and_A_DF = pd.DataFrame(columns=['Questions','Answers','b','c','v'])
    prepared_Q_and_A_DF[['prefix', 'b', 'cv']] = Q_and_A_DF['Answers'].str.extract(r'(\d+\s)?(\w+) (\d+:\d+)')
    prepared_Q_and_A_DF['b_full'] = prepared_Q_and_A_DF['prefix'].fillna('') + prepared_Q_and_A_DF['b'].fillna('')
    prepared_Q_and_A_DF['b'] = prepared_Q_and_A_DF['b_full']
    prepared_Q_and_A_DF = prepared_Q_and_A_DF.drop('b_full', axis=1)
    prepared_Q_and_A_DF[['c', 'v']] = prepared_Q_and_A_DF['cv'].str.split(':', expand=True)
    prepared_Q_and_A_DF = prepared_Q_and_A_DF.drop(['prefix', 'cv'], axis=1)
    prepared_Q_and_A_DF['Questions'] = Q_and_A_DF['Questions'].str.replace(r'^\d+\.\s', '', regex=True)
    prepared_Q_and_A_DF['Answers'] = Q_and_A_DF['Answers'].str.replace(r'^\d+\.\s', '', regex=True)
    prepared_Q_and_A_DF['Answers'] = prepared_Q_and_A_DF['Answers'].str.replace(r'\s\(.+?\)', '', regex=True)
    prepared_Q_and_A_DF = prepared_Q_and_A_DF[prepared_Q_and_A_DF['b'] != '']
    book_mapping_DF = LoadDF(book_mapping_csv_filepath)
    book_mapping_dict = book_mapping_DF.set_index('SF')['LF'].to_dict()
    prepared_Q_and_A_DF['b'] = prepared_Q_and_A_DF['b'].replace(book_mapping_dict)
    ESV_Bible_Book_id_dict = ESV_Bible_Book_id_DF.set_index('b')['id'].to_dict()
    prepared_Q_and_A_DF['b'] = prepared_Q_and_A_DF['b'].replace(ESV_Bible_Book_id_dict)
    prepared_Q_and_A_DF.fillna('', inplace=True)
    prepared_Q_and_A_DF['b'] = prepared_Q_and_A_DF['b'].astype(int)
    prepared_Q_and_A_DF['c'] = prepared_Q_and_A_DF['c'].astype(int)
    prepared_Q_and_A_DF['v'] = prepared_Q_and_A_DF['v'].astype(int)
    ESV_Bible_DF['b'] = ESV_Bible_DF['b'].astype(int)
    ESV_Bible_DF['c'] = ESV_Bible_DF['c'].astype(int)
    prepared_Q_and_A_DF = pd.merge(prepared_Q_and_A_DF,ESV_Bible_DF,on=['b', 'c', 'v'])
    prepared_Q_and_A_DF.to_csv(prepared_Q_and_A_DF_filepath, index=False)
    return prepared_Q_and_A_DF

#CreateOrLoadTokenizedData Function
def CreateOrLoadTokenizedData(Q_and_A_DF,ESV_Bible_DF,ESV_Bible_Book_id_DF,book_mapping_csv_filepath,prepared_Q_and_A_DF_filepath,create_or_load_string='load'):
    if create_or_load_string in ['Create','create']:
        prepared_Q_and_A_DF = CreatePreparedQAndAData(Q_and_A_DF,ESV_Bible_DF,ESV_Bible_Book_id_DF,book_mapping_csv_filepath,prepared_Q_and_A_DF_filepath)
    else:
        prepared_Q_and_A_DF = LoadDF(prepared_Q_and_A_DF_filepath)
    return prepared_Q_and_A_DF
    
#################################################################################################################
### Main Functions

if __name__ == "__main__":

    #Variables
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    initial_Q_and_A_DF_filepath = 'Initial_Data/QandA_Dataset.csv'
    ESV_Bible_Book_id_DF_filepath = 'Additional_Data/ESV_Bible_Book_id_DF.csv'
    book_mapping_csv_filepath = 'Additional_Data/ESV_Bible_Book_SF_To_LF.csv'
    prepared_Q_and_A_DF_filepath = 'Additional_Data/Prepared_QandA_Dataset.csv'
    create_or_load_string = 'Create'
    
    #Main Code
    ESV_Bible_DF = LoadDF(ESV_Bible_DF_filepath)
    Q_and_A_DF = LoadDF(initial_Q_and_A_DF_filepath)
    ESV_Bible_Book_id_DF = LoadDF(ESV_Bible_Book_id_DF_filepath)
    prepared_Q_and_A_DF = CreateOrLoadTokenizedData(Q_and_A_DF,ESV_Bible_DF,ESV_Bible_Book_id_DF,book_mapping_csv_filepath,prepared_Q_and_A_DF_filepath,create_or_load_string)
    
    