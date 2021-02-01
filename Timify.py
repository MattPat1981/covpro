import pandas as pd
import numpy as np



def timify(data):
    '''
    returns a usable national dataframe using inputs from the JHU timeseries data set
    data is a pandas dataframe
    '''
    # create a county_state column
    df = data.copy()
    county_state = df['Admin2'] + " County, " + df['Province_State']
    county_state = pd.DataFrame(county_state)
    county_state.columns = ['county_state']

    # remove redundant columns
    remove_cols = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2',
                     'Province_State','Country_Region', 'Lat', 'Long_',
                     'Combined_Key']
    if 'Population' in df.columns: remove_cols.append('Population')
    df.drop(columns=remove_cols, inplace=True)

    #place the county_state column in the front
    df = county_state.merge(df, how='outer', left_index=True, right_index=True)

    return df

def localize(data, topic, county_state):
    '''
    returns a localized timeseries dataframe using inputs from timify function
    df           is a pandas dataframe
    topic        is a string, either cases or deaths for the dataframe
    county_state is a string, the county and state of the time series dataframe
    '''
    df = data.copy()
    #print('TEST 1: using a copy of the passed dataframe')
    # Break the needed data row away from the larger dataframe
    #print(f'TEST 2: creating a copy of the df using {county_state}')
    local = df[df['county_state'] == county_state]
    #print(f'TEST 2 COMPLETE')
    # Make the county_state the index
    #print(f'TEST 3: set index to {county_state}')
    local.set_index('county_state', inplace=True)
    #print('TEST 3 COMPLETE')
    # use df.loc to pull the pd.Series, flipping the axes, then make it a df again
    #print('TEST 4: use df.loc to pull the pd.Series, flipping the axes, then make it a df again')
    local = pd.DataFrame(local.loc[county_state])
    #print('TEST 4 COMPLETE')
    # Create a date column in order to change types
    # (this is a hacky way to do it)
    local['date'] = local.index
    # convert date column to datetime type
    local['date'] = pd.to_datetime(local['date'])
    # make the datetime column the index
    local.set_index(['date'], inplace=True)
    # change the column name to the desired topic of information
    local.columns=[topic]

    return local

