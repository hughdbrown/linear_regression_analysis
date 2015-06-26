#!/usr/bin/env python
from __future__ import print_function, division

from operator import itemgetter
import os.path

import wget

import pandas as pd
import numpy as np
import statsmodels.api as sm


URL = 'https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv'
FILE = 'loansData.csv'


def load_data():
    if not os.path.exists(FILE):
        print("Downloading data from url")
        wget.download(URL)
    print("Loading data from file")
    return pd.read_csv(FILE)


def clean_data(df):
    print("Cleaning data")
    df['Interest.Rate'] = df['Interest.Rate'].str.replace('%', '').astype('float')
    df['Loan.Length'] = df['Loan.Length'].str.replace(' months', '').astype('int')
    df['FICO.Score'] = df['FICO.Range'].str.split('-').apply(itemgetter(0)).astype('int')


def linear_regression(loansData):
    print("Extracting series for regression")
    intrate = loansData['Interest.Rate']
    loanamt = loansData['Amount.Requested']
    fico = loansData['FICO.Score']

    # The dependent variable
    print("Creating dependent variable")
    y = np.matrix(intrate).transpose()

    # The independent variables shaped as columns
    print("Creating independent variables")
    x1 = np.matrix(fico).transpose()
    x2 = np.matrix(loanamt).transpose()

    # Put the two columns together to create an input matrix 
    print("Stacking independent variables")
    x = np.column_stack([x1, x2])

    # Create a linear model
    print("Creating linear model")
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    f = model.fit()
    
    print(dir(f))
    
    # Print results
    print("Printing results")
    print(f.summary())


def main():
    loansData = load_data()
    clean_data(loansData)
    linear_regression(loansData)


if __name__ == '__main__':
    main()
