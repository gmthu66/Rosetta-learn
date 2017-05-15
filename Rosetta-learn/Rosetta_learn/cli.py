import click
import os, sys
import errno
import pandas as pd
import numpy as np
import proteinmodel as dm
from sklearn.preprocessing import MinMaxScaler



all_colors = 'black', 'red', 'green', 'yellow', 'blue', 'magenta', \
             'cyan', 'white'

@click.command()
@click.argument('input_file')
@click.option('--model', '-m', type=click.Path(), help='An existing dnamodel .h5 file')


def main(input_file, model):
    """Automated, experimentally-driven design and optimization of protein sequences."""
    greet = 'Hello'


    raw_data = pd.read_excel(input_file, header=0)

    #first column is the sequence, last column is the output, all cols in between are metrics
    col_names = list(raw_data.columns)
    col_names[0] = u'sequence'
    col_names[-1] = u'output1'
    raw_data.columns = col_names
    

    #TO DO: should do this for each column
    df = raw_data[np.isfinite(raw_data['output1'])]
    df = df.set_index('sequence')

    #data cleaning needs to be done:
        #remove any columns with low variability
        #then shuffle the data
    print df.shape

    #rescaling input to have mean 0 variance 1 - for efficient backprop
    scaler = MinMaxScaler()
    new_input_df = scaler.fit_transform(df)
    new_input_df = pd.DataFrame(data=new_input_df, index=df.index, columns=df.columns)
    #new_input_df.hist(figsize=(30, 25))

    #eliminating cols with lack of variability
    new_input_df = new_input_df.drop(new_input_df.columns[new_input_df.var() < 0.002], axis=1)
    #new_input_df.hist(figsize=(30, 25))


    #shuffle the dataframe here
    df = new_input_df.sample(frac=1)


    #print new_input_df.head()

    print df.shape

    dnaCNN = dm.proteinModel(df, filename=model) 
    
    dnaCNN.train()
    
    # dnaCNN.design()

    # dnaCNN.save()

    # dnaCNN.test()
