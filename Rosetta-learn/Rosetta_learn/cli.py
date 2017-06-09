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
    print raw_data.columns
    

    #TO DO: should do this for each column
    df = raw_data[np.isfinite(raw_data['output1'])]
    #shuffle the dataframe here - have to do this here when sequences are a column
    df = df.sample(frac=1)
    #ADD BACK
    df = df.set_index('sequence') 

    #data cleaning needs to be done:
        #remove any columns with low variability
        #then shuffle the data
    #print df.head()

    #rescaling input to have mean 0 variance 1 - for efficient backprop
    scaler = MinMaxScaler()
    new_input_df = scaler.fit_transform(df)
    new_input_df = pd.DataFrame(data=new_input_df, index=df.index, columns=df.columns)
    
    #eliminating cols with lack of variability
    #ADD BACK
    print new_input_df.shape
    fin_input_df = new_input_df

    #ADD BACK
    #fin_input_df = fin_input_df.drop(fin_input_df.columns[fin_input_df.var() < 0.002], axis=1)

    print fin_input_df.shape
    fin_input_df['output1'] = new_input_df['output1']
    print fin_input_df.columns


    dnaCNN = dm.proteinModel(fin_input_df, filename=model) 
    
    dnaCNN.train()
    
    # dnaCNN.design()

    dnaCNN.save()

    # dnaCNN.test()
