# Steinbeck.py is a python program designed specifically for
# pulling time series data from the Johns Hopkins University
# COVID-19 github and turning them in to usable timeseries
# csv's for analysis.

# @author Matt Paterson, hello@HireMattPaterson.com
# This program written and produced for and by Cloud Brigade


import pandas as pd
import numpy as np
import datetime
import boto3
from io import StringIO
BUCKETNAME = 'covid-v1-part-3-data-bucket'

def timify(data):
    '''
    returns a usable national dataframe using inputs from the JHU timeseries data set,
            and second item it returns is a dataframe with the population of each county
    data is a pandas dataframe
    '''
    # create a county_state column
    df = data.copy()
    county_state = df['Admin2'] + " County, " + df['Province_State']
    county_state = pd.DataFrame(county_state)
    county_state.columns = ['county_state']

    # create a population flag to tell the function whether or not to return
    # a population dataframe too
    pop_flag=False
    # remove redundant columns
    remove_cols = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2',
                     'Province_State','Country_Region', 'Lat', 'Long_',
                     'Combined_Key']
    df_2 = df.copy()
    if 'Population' in df.columns:
        pop_flag=True
        remove_cols.append('Population')
        df_2 = pd.merge(county_state, df_2['Population'].to_frame(), how='outer', left_index=True, right_index=True)
        df_2 = df_2.set_index('county_state')
    df.drop(columns=remove_cols, inplace=True)

    #place the county_state column in the front
    df = county_state.merge(df, how='outer', left_index=True, right_index=True)

    if pop_flag: return df, df_2
    return df

def localize(data, topic, county_state):
    '''
    returns a localized timeseries dataframe using inputs from timify function
    df           is a pandas dataframe
    topic        is a string, either cases or deaths for the dataframe
    county_state is a string, the county and state of the time series dataframe
    '''
    df = data.copy()

    # Break the needed data row away from the larger dataframe
    local = df[df['county_state'] == county_state]

    # Make the county_state the index
    local.set_index('county_state', inplace=True)

    # use df.loc to pull the pd.Series, flipping the axes, then make it a df again
    local = pd.DataFrame(local.loc[county_state])

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


def pull_raw(datapath='/home/ubuntu/covpro/data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/'): 
    '''
    returns two dataframes, cases and deaths in that order, from the JHU git repo
    datapth is the path on the local machine to the daily updated JHU git repo
    '''
    # Match destination files to variables
    case_ts = 'time_series_covid19_confirmed_US.csv'
    death_ts = 'time_series_covid19_deaths_US.csv'
    # Read in the confirmed_cases and deaths data sets
    #     cases = pd.read_csv(datapath + case_ts)
    #     deaths = pd.read_csv(datapath + death_ts)
    return pd.read_csv(datapath + case_ts), pd.read_csv(datapath + death_ts)


def national_update( datapath, output_path, bucket=BUCKETNAME):
    '''
    national_update() function
    returns a dictionary containing covid-19 time series dataframes for each US county
    datapath is a string, the path to the JHU data dump from github
    output_path is a string, the path for where to send this file
    bucket   is a string, the s3 bucket where the data is sent
    '''
    # Pull the latest update from JHU and timify the data
    cases, deaths = pull_raw(datapath)
    cases = timify(cases)
    deaths, population = timify(deaths)

    # Create an empty dictionary to fill
    national = {}
    # setting a counter
    i=0
    print('Building dictionary or sending csv files...')
    # Iterate through the cases dataframe that was built above
    for county in deaths['county_state']:
        # Skip the entry if the datatype is not correct in the df
        if type(county) == str:
            # Skip the entry if it's a state summary column
            if 'Out of' not in county:
                i+=1
                #
                # Create dataframes for cases and deaths using sbk.localize
                df_cases =  localize(cases, 'cases', county)
                df_deaths = localize(deaths, 'deaths', county)
                # Merge these dataframes
                df = pd.merge(df_cases, df_deaths, left_index=True, right_index=True)
                # Add columns for Month, new daily cases, and new daily deaths
                df['Month'] = df.index.month
                df['prior_day_cases'] = df['cases'].shift(1).copy()
                df['prior_day_deaths'] = df['deaths'].shift(1).copy()
                df['new_daily_cases'] = df['cases'] - df['prior_day_cases'].copy()
                df['new_daily_deaths'] = df['deaths'] - df['prior_day_deaths'].copy()
                df.drop(columns=(['prior_day_cases', 'prior_day_deaths']), inplace=True)
                # Add columns for cases and deaths per 100K as well as weekly new deaths per 100K
                df['cases_per_100K'] = round((df['cases'] / population.loc[county]['Population']) * 100_000, 2)
                df['deaths_per_100K'] = round((df['deaths'] / population.loc[county]['Population']) * 100_000, 2)

                county_linked = county.replace(" ", "_")
                county_linked = county_linked.replace(",","")

                #Send to csv in repo
                csv_buffer = StringIO()
                df.to_csv(csv_buffer)
                s3_resource = boto3.resource('s3')
                s3_resource.Object(bucket, f'{output_path}/{county_linked}.csv').put(Body=csv_buffer.getvalue())

                #df.to_csv(output_path + county_linked + '.csv')
                national[county] = df
                if i % 100 == 0:
                    print(f'{i} counties updated and sent to s3')

    print(f'SUCCESS! Created a dictionary with {i} key-value pairs')
    return national


def build_county(county):
    '''
    returns a pandas dataframe object with the up-to-date time-series data
            for the COVID-19 pandemic as pulled from the Johns Hopkins github
    params:
        county    a string in format '<county_name> County, <state_name>'
    '''
    # Set the data path variables for input and output
#     output_path = '~/Dropbox/Business/CloudBrigade/Projects/Covid-part-six/data/'
#     datapath = '~/Dropbox/School/GeneralAssembly/Projects/COVID-19/csse_covid_19_data/csse_covid_19_time_series/'

#     # Match destination files to variables
#     case_ts = 'time_series_covid19_confirmed_US.csv'
#     death_ts = 'time_series_covid19_deaths_US.csv'

#     # Read in the confirmed_cases and deaths data sets
#     cases = pd.read_csv(datapath + case_ts)
#     deaths = pd.read_csv(datapath + death_ts)

    # Convert each data set into usable pandas dataframes
    cases, deaths = pull_raw()
    cases = timify(cases)
    deaths, population = timify(deaths)

    # Create a new time series data set for Santa Cruz County, California

    df_cases =  localize(cases, 'cases', county)
    df_deaths = localize(deaths, 'deaths', county)

    # Merge the two dataframes to create one Santa Cruz time series df
    #cty = pd.merge(df_cases, df_deaths, left_index=True, right_index=True)

    # This block Sends it to a new csv file in the output datapath
    #cty.to_csv(output_path + 'historical/' + 'santa_cruz_historical.csv')

    # This block Sends it to a daily update csv file in the output datapath
    #cty.to_csv(output_path + 'live/' + str(datetime.datetime.now())[:10] + '.csv')

    #return cty
    return pd.merge(df_cases, df_deaths, left_index=True, right_index=True)

def graph_county(county, topic=None):
    '''
    returns a dataframe and prints a map of the deaths
    county is a string, the '<county_name> County, <state_name>'
    topic  is a string, either 'cases' or 'deaths', default is 'both'
    '''
    import matplotlib.pyplot as plt
    df = build_county(county)

    if topic:
        plt.figure(figsize=(15, 5))
        plt.plot(df[topic])
        plt.title(f'COVID-19 {topic} in {county}', fontsize=36);
    else:
        plt.figure(figsize=(15, 5))
        plt.plot(df['cases'])
        plt.title(f'COVID-19 cases in {county}', fontsize=36);
        plt.figure(figsize=(15, 5))
        plt.plot(df['deaths'])
        plt.title(f'COVID-19 deaths in {county}', fontsize=36);

    return df


def get_county(county_name, datapath='~/Dropbox/School/GeneralAssembly/Projects/COVID-19/csse_covid_19_data/csse_covid_19_time_series/'):
    '''
    returns a fully formed dataframe for the requested county_name
    county_name   is a string in format '<county> County, <state>'
    '''
    county_linked = county_name.replace(" ", "_")
    county_linked = county_linked.replace(",","")

    return pd.read_csv(datapath+county_linked+'.csv')

