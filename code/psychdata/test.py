import os
import psych
import numpy as np

psychdata = psych.Psychdata()
data = psychdata.get_corr()


print(data)

print('This is a', data.shape ,'Correlation matrix! For explanation of the data see\nhttps://www.personality-project.org/r/html/bfi.html \n')
