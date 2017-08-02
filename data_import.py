'''
Provides functionalities for importing the challenge data.
'''

import os
import zipfile

import pandas as pd

DEFAULT_KGGLCNCR_DATA_PATH = '/share/projects/TaskForce_ML/kaggle_personalized_medicine/data'
#DEFAULT_KGGLCNCR_DATA_PATH = '/home/sebastian/prjcts/kgglcncr/data'

################################################################################
# Import text data #############################################################
################################################################################

def import_text_data_as_dataframe(path):
    '''
    Read challenge text data, i.e. data organized like
    'training_text' and 'test_text', into RAM as Pandas
    dataframe.

    Args:
        path (str): The path of the zip file containing the text data

    Returns:
        pd.DataFrame: Pandas dataframe with ID and Text column
    '''
    # taken from notebooks/kagglePersonalizedMedicine_01_importToPandas
    return pd.read_csv("../data/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

def import_text_data_as_generator(path):
    '''
    Read challenge text data, i.e. data organized like
    'training_text' and 'test_text', as Python generator.

    Args:
        path (str): The path of the zip file containing the text data

    Returns:
        TextDataGenerator: Python generator yielding (ID, text) tuples
    '''

    class TextDataGenerator(object):
        def __init__(self, path, name_of_file):
            self.path = path
            self.name_of_file = name_of_file
        def __iter__(self):
            with zipfile.ZipFile(self.path) as zipf:
                with zipf.open(self.name_of_file, 'r') as f:
                    _ = f.readline() # header
                    for line in f:
                        ID, text = line.split(b'||')
                        ID, text = int(ID), text.decode('utf-8').strip()
                        yield ID, text

    name_of_file = os.path.splitext(os.path.basename(path))[0]
    return TextDataGenerator(path, name_of_file)

def import_text_data(path, mode='in_memory'):
    '''
    Wrapper function for the different modes of text data import.

    Args:
        path (str): The path of the zip file containing the text data
        mode (str): Import mode, either 'in_memory' or 'stream'

    Returns:
        If mode equals 'in_memory' you get a Pandas dataframe with an ID
        and a text column. If mode equals 'stream' you get a Python generator
        yielding the (ID, text) tuples
    '''
    if mode == 'in_memory':
        return import_text_data_as_dataframe(path)
    elif mode == 'stream':
        return import_text_data_as_generator(path)

def import_training_text(data_path = DEFAULT_KGGLCNCR_DATA_PATH, mode='in_memory'):
    '''
    Import the training text data.

    Args:
        data_path (str): Path of directory where the challenge data files are located in
        mode (str): Import mode, either 'in_memory' or 'stream'

    Returns:
        The training text data in the format specified by the mode-argument.
        See 'import_text_data'.
    '''
    return import_text_data(os.path.join(data_path, 'training_text.zip'), mode)

def import_test_text(data_path = DEFAULT_KGGLCNCR_DATA_PATH, mode='in_memory'):
    '''
    Import the test text data.

    Args:
        data_path (str): Path of directory where the challenge data files are located in
        mode (str): Import mode, either 'in_memory' or 'stream'

    Returns:
        The test text data in the format specified by the mode-argument.
        See 'import_text_data'.
    '''
    return import_text_data(os.path.join(data_path, 'test_text.zip'), mode)

################################################################################
# Import variant data ##########################################################
################################################################################

def import_training_variants(data_path = DEFAULT_KGGLCNCR_DATA_PATH):
    '''
    Import the training variant data.

    Args:
        data_path (str): Path of directory where the challenge data files are located in

    Returns:
        pd.DataFrame: The training variant data.
    '''
    return pd.read_csv(os.path.join(data_path, 'training_variants.zip'), compression='zip')

def import_test_variants(data_path = DEFAULT_KGGLCNCR_DATA_PATH):
    '''
    Import the test variant data.

    Args:
        data_path (str): Path of directory where the challenge data files are located in

    Returns:
        pd.DataFrame: The test variant data.
    '''
    return pd.read_csv(os.path.join(data_path, 'test_variants.zip'), compression='zip')
