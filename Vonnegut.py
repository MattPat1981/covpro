# Vonnegut.py is a python program designed specifically for
# pulling time series data from the Johns Hopkins University
# COVID-19 github and turning them in to usable timeseries
# csv's for analysis. This program sends files to an s3 bucket
# solely for Santa Cruz, California

# @author Matt Paterson, hello@HireMattPaterson.com
# This program written and produced for and by Cloud Brigade


import pandas as pd
import numpy as np
import datetime
import boto3
from io import StringIO

from Timify import timify, localize
bucket = 'covid-v1-part-3-data-bucket'


# Set the data path variables for input and output
output_path = '/home/ubuntu/covpro/timeseries/'
datapath = '/home/ubuntu/covpro/data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/'

# Match destination files to variables
case_ts = 'time_series_covid19_confirmed_US.csv'
death_ts = 'time_series_covid19_deaths_US.csv'

# Read in the confirmed_cases and deaths data sets
cases = pd.read_csv(datapath + case_ts)
deaths = pd.read_csv(datapath + death_ts)


# Convert each data set into usable pandas dataframes
cases = timify(cases)
deaths = timify(deaths)

# Create a new time series data set for Santa Cruz County, California

df_cases =  localize(cases, 'cases', 'Santa Cruz County, California')
df_deaths = localize(deaths, 'deaths', 'Santa Cruz County, California')

# Merge the two dataframes to create one Santa Cruz time series df
sc_ts = pd.merge(df_cases, df_deaths, left_index=True, right_index=True)

# Send it to a new csv file in the output datapath
#sc_ts.to_csv(output_path + 'historical/' + 'santa_cruz_historical')

# Send it to a daily update file in the output datapath
#sc_ts.to_csv(output_path + 'live/' + str(datetime.datetime.now())[:10] + '.csv')
csv_buffer = StringIO()
sc_ts.to_csv(csv_buffer)

s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'timeseries/live/' + str(datetime.datetime.now())[:10] + '/data.csv').put(Body=csv_buffer.getvalue())

print(f'Sent file to s3://{bucket}/timeseries/live/{str(datetime.datetime.now())[:10]}/data.csv')


