import pandas as pd # Use pandas for handling missing entries "NA" in bfi data 
import numpy as np


class Psychdata:
    def __init__(self):
        self.df= pd.read_csv("bfi.csv")
        
    def get_raw_data(self):
        return self.df
        
    def get_reduced_data(self):
        return self.df.drop(['Unnamed: 0','gender', 'education', 'age'],axis=1,inplace=True)
    
    def get_corr(self):
        data = self.get_reduced_data()
        corrmatrix=self.df.corr(method='pearson', min_periods=1)
        return corrmatrix.to_numpy()
