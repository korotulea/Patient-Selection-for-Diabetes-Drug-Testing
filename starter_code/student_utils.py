import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df_new = pd.merge(df, ndc_df[['NDC_Code', 'Proprietary Name']], left_on='ndc_code', right_on='NDC_Code')
    df_new['generic_drug_name'] = df_new['Proprietary Name']
    df_new = df_new.drop(['NDC_Code', 'Proprietary Name'], axis=1)
    return df_new

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_df = df.sort_values(['patient_nbr', 'encounter_id']).groupby('patient_nbr').first().reset_index()
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    uniq_patient_id = df[patient_key].unique()
    train_p_id, test_valid_p_id, _, _ = train_test_split(uniq_patient_id, uniq_patient_id, test_size=0.4, random_state=0)
    test_p_id, valid_p_id, _, _ = train_test_split(test_valid_p_id, test_valid_p_id, test_size=0.5, random_state=0)
    train = df[df[patient_key].isin(train_p_id)]
    validation = df[df[patient_key].isin(valid_p_id)]
    test = df[df[patient_key].isin(test_p_id)]
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_cat_colum = tf.feature_column.categorical_column_with_vocabulary_file(c, vocab_file_path, \
                                                                          num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_cat_colum)        
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    zscore_norm = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(col, default_value=default_value, \
                                                          normalizer_fn=zscore_norm, \
                                                          dtype=tf.float64)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col, threshold=5):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    threshold=5 is default
    return:
        binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    binary_prediction = df[col].apply(lambda x: 1 if x >= threshold else 0)
    return binary_prediction
