import pandas as pd

def load_auto(file_path):

	# import data
	Auto = pd.read_csv(filepath_or_buffer=file_path, na_values='?', dtype={'ID': str}).dropna().reset_index()
	Auto.columns
	# Extract relevant input and output for traning
	X_train = Auto.drop(columns='mpg')
	Y_train = Auto[['mpg']]

	return X_train, Y_train