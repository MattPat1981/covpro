# Import the needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# For linear regression function
from sklearn.ensemble          import GradientBoostingRegressor
from sklearn.linear_model      import LogisticRegression, LinearRegression
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import r2_score, mean_squared_error

# For clustering function
from sklearn.cluster           import KMeans
from sklearn.decomposition     import PCA
from sklearn.preprocessing     import StandardScaler, OneHotEncoder, PolynomialFeatures

import boto3
from io import StringIO


import warnings
warnings.filterwarnings("ignore")

BUCKETNAME = 'covid-v1-part-3-data-bucket'

def report_date(lag=1):
    '''
    returns a string representing the date for the report we are running
    lag is an int, representing how many days in the past to run the report,
        the default lag is 1 days
    '''
    today=str(datetime.datetime.now())[:8]
    date = int(str(datetime.datetime.now())[8:10])
    if date > lag:
        day = date - lag
        zero = str(0)
        if day < 10:
            return f'{today}{zero}{day}'
        return f'{today}{day}'

    # If the date is the first of the month, take the report
    # for the last day of the prior month
    month = str(int(today[5:7]) - 1)
    year = today[:4]

    thirty_one = ['01', '03', '05', '07', '08', '10', '12']
    thirty     = ['04', '06', '09', '11']

    if month in thirty_one:
        day = '31'
    elif month in thirty:
        day = '30'
    else:
        day = '28'
    return f'{year}-{month}-{day}'

def jhu_date(date):
    '''
    returns a string representing the date for the report we are running
    date is a string representing the date found by report_date()
    '''
    year = date[:4]
    month = date[5:7]
    day = date[8:10]
    zero = 0
    #if int(day) < 10:
    #    return f'{month}-{zero}{day}-{year}'
    return f'{month}-{day}-{year}'


def daily_snapshot(datapath, covpath, date, bucket=BUCKETNAME):
    '''
    sends the daily snapshot of the covid totals incorporated in to the
          static data set to a data folder to be used by the models
    datapath is a string, the relative path through the s3 bucket
    covpath  is a string, the relative or absolute path the the daily
             updated covid-19 data from the Johns Hopkins github repo
    date     is a string representing the report date
    bucket   is a string, the name of the s3 bucket for storage in aws
    '''
    # set filename for the daily data
    cov_date = jhu_date(date)

    filename = f'{covpath}/{cov_date}.csv'
    print(f'searching {covpath} for {cov_date}.csv')
    try:
        # create new dataframe with the covid data
        covid = pd.read_csv(filename)
        print(f'SUCCESS, creating "covid" dataframe  using {filename}')
    except:
        print(f'Exception: {filename} not found. Check your file location or try to run prior day\'s report')

    # strip down the covid data to cases and deaths
    covid = covid[covid['Country_Region']=='US']
    covid['county_state'] = covid['Admin2'] + ' County, ' + covid['Province_State']
    covid.set_index('county_state', inplace=True)
    covid.rename(columns=({
        'Confirmed': 'confirmed_cases',
        'Deaths': 'deaths'
    }), inplace=True)
    covid=covid[['confirmed_cases', 'deaths']]

    # pull in prior collective data set
    filename = 'cov_soc_eco.csv'

    ### DELETE THIS PRINT STATEMENT UPON COMPLETION###

    print(f'searching {datapath} for {filename}')
    # create new dataframe with the covid data
    full = pd.read_csv(f's3://{bucket}/{datapath}/{filename}')
    print(f'SUCCESS, creating dataframe using {filename}')

    # set the index in the dataframe to 'county_state'
    full['county_state'] = full['county'] + ", " + full['state']
    full.set_index('county_state', inplace=True)

    ### This part will vary **********************
    # set the index in the covid dataframe for consistency

    # print the cases and deaths numbers from the old data_set
    print("Think about changing this to be yesterday's numbers:")
    print('cases total on prior data set: ', full['confirmed_cases'].sum())
    print('deaths total on prior data set: ', full['deaths'].sum())

    # open the dataframe confirmed cases and deaths columns for replacement
    full.drop(columns=['confirmed_cases', 'deaths'], inplace=True)

    # merge the dataframes
    full = full.merge(covid, how='left', left_on=full.index, right_on=covid.index)
    full['county_state'] = full['key_0']
    full.drop(columns='key_0', inplace=True)
    full.set_index('county_state', inplace=True)

    # print the cases and deaths numbers from the new data_set
    print("Think about changing this to be today's numbers:")
    print('cases total on new data set: ', full['confirmed_cases'].sum())
    print('deaths total on new data set: ', full['deaths'].sum())

    # overwrite the covid-per-capita columns
    full['cases_per_100k'] = (full['confirmed_cases'] / full['Population']) * 100_000
    full['deaths_per_100k']= (full['deaths'] / full['Population']) * 100_000

    # export the new dataframe to csv
    
    csv_buffer = StringIO()
    full.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, f'{datapath}/updated_snapshot.csv' ).put(Body=csv_buffer.getvalue())

    print('SUCCESS, sending file updated_snapshot.csv to data folder')

    print('\nDaily Snapshot run, commence Machine Learning models\n')


def show_rmse(model_1, model_2):
    '''
    returns 1 for error, 0 for complete
            prints the Root Mean Squared error of one or two models
    '''
    try:
        print('Root Mean Squared Error of the LR: ',(mean_squared_error(y_test, model_1.predict(X_test)))**(1/2))
        print('Root Mean Squared Error of the GB: ',(mean_squared_error(y_test, model_2.predict(X_test)))**(1/2))
        return 0
    except:
        print('An exception occurred')
        return 1

def i_regress(dataframe, features, target, test_size=0.2, n_estimators=100):
    '''
    returns two models, a LinearRegression() and GradientBoostingRegressor(),
            in that order, and also prints out the r2 scores of each
    dataframe  is the dataframe being used for the testing
    features   is a list of numeric data
    target     is a string, the column name from the dataframe of the target
    test_size  is a float between 0.0 and 1.0 exclusive used
               in the train_test_split() function
    n_estimators is an int used in the GradientBoostingRegressor()
    '''

    # import librarires if exporting this function to a useful library

    lr = LinearRegression()
    gb = GradientBoostingRegressor(n_estimators=n_estimators)

    X = dataframe[features]
    y = dataframe[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                       random_state=42)
    lr.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    print(f'Models successfully built: Target = "{target}"')
    print('*'*30)
    print('Training Scores: ')
    print('LinearRegression         : ', lr.score(X_train, y_train))
    print('GradientBoostingRegressor: ', gb.score(X_train, y_train))
    print('*'*30)
    print('Testing Scores: ')
    print('LinearRegression         : ', lr.score(X_test, y_test))
    print('GradientBoostingRegressor: ', gb.score(X_test, y_test))

    print('*'*30)
    show_rmse(lr, gb)

    print('\n')

    return lr, gb

def run_regression_models(datapath, filename, date, model=False, bucket=BUCKETNAME):
    '''
    Runs Linear Regression models and sends a new 85_col.csv to data storage
    datapath is a string, the relative or absolute path to the data storage
    filename is the name of the daily snapshot file
    date     is a string representing the report date
    bucket   is a string representing the s3 bucket name
    '''
    # call the pandas read_csv() function to create a dataframe
    covid_df = pd.read_csv(f's3://{bucket}/{datapath}/{filename}')
    covid_df.set_index('county_state', inplace=True)

    # use one-hot-encoding to split up the states
    covid_df = pd.get_dummies(columns=['state_abr'], data=covid_df, prefix="", prefix_sep="")

    # create a features list
    cols = dict(covid_df.dtypes)
    features = []
    for col in cols:
        if "object" not in str(cols[col]):
            features.append(col)

    # export this dataset for use in the KMeans model:
    csv_buffer = StringIO()
    covid_df[features].to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, f'{datapath}/85_cols.csv').put(Body=csv_buffer.getvalue())

    #commenting out this next bit as changing to an s3 StringIO upload
    #covid_df[features].to_csv(datapath+'85_cols.csv')

    # Remove the target columns from the features list
    features.remove('deaths')
    features.remove('confirmed_cases')
    features.remove('deaths_per_100k')
    features.remove('cases_per_100k')
    targets_list = ['deaths', 'confirmed_cases', 'deaths_per_100k', 'cases_per_100k']

    # If the model input is envoked, run models:
    if model:

        # call the i_regress() function to run the models themselves
        lr_1, gb_1 = i_regress(covid_df, features, target='deaths', test_size=.3, n_estimators=500)
        lr_2, gb_2 = i_regress(covid_df, features, target='confirmed_cases', test_size=.2, n_estimators=200)
        lr_3, gb_3 = i_regress(covid_df, features, target='deaths_per_100k', test_size=.4, n_estimators=500)
        lr_4, gb_4 = i_regress(covid_df, features, target='cases_per_100k', test_size=.35, n_estimators=500)

    ## Begin re-working the dataframe for exporting to Tableau Dashboard
    # Receate a features list containing only the numerical features
    cols = dict(covid_df.dtypes)
    features = []
    for col in cols:
        if "object" not in str(cols[col]) and col not in targets_list:
            features.append(col)
    # Create a list of the state columns and seperate them from the features list
    states = []
    for feat in features:
        if len(feat)==2 and 'Q' not in feat:
            states.append(feat)
    for state in states:
        features.remove(state)
    # create a locations list
    locations = ['latitude', 'longitude', 'fips']

    for loc in locations:
        features.remove(loc)
    print('For re-assessment, wittling down to ', len(features), ' features')

    # If the model input is envoked, run models:
    if model:

        # Run the models again and look for movement
        print('\nRe-running the models using new features\n')
        lr_5, gb_5 = i_regress(covid_df, features, target='deaths', test_size=.3, n_estimators=500)
        lr_6, gb_6 = i_regress(covid_df, features, target='confirmed_cases', test_size=.2, n_estimators=200)
        lr_7, gb_7 = i_regress(covid_df, features, target='deaths_per_100k', test_size=.4, n_estimators=500)
        lr_8, gb_8 = i_regress(covid_df, features, target='cases_per_100k', test_size=.35, n_estimators=500)

        print('\nIn next iteration, this is a good place for a printout of the')
        print('  results of the modeling. Show a graph demonstrating changes\n')

    strongest = ['deaths_per_100k', 'cases_per_100k', 'pct_white',
                 'pct_black', 'percapita_income', 'median_household_income',
                 'median_family_income']

    plt.figure(figsize=(18, 12))


    #sns.set(font_scale=3) # font size 2
    sns.heatmap(covid_df[strongest].corr(), cmap='coolwarm', annot=True );
    plt.title('Heatmap of strongest correlations in Model', fontsize=24)

def prep_final_data(datapath, dashboard_datapath, archive_path, 
        filename, date, bucket=BUCKETNAME):
    '''
    sends final data set to storage
    datapath is a string, the relative or absolute path to the data storage
    dashboard_datapath is a string, the path to where dashboard file is stored
    archive_path is a string, the path to where the archvies are stored
    filename is the name of the daily snapshot file
    date     is a string representing the report date
    bucket   is a string representing the name of the s3 bucket
    '''
    print(f'\nBeginning final data set formation for {date}')

    # read in the csv to build dataframe
    covid_df = pd.read_csv(f's3://{bucket}/{datapath}/{filename}')
    covid_df.set_index('county_state', inplace=True)

    # drop the state columns
    df_lower = covid_df[['confirmed_cases', 'deaths', 'latitude', 'longitude', 'fips',
       'percapita_income', 'median_household_income', 'median_family_income',
       'number_of_households', 'Population', 'pct_white', 'pct_black',
       'pct_asian', 'pct_hispanic', 'pct_native_american', 'pct_hawaiian',
       'QMB_Only', 'QMB_plus_Full', 'SLMB_only', 'SLMB_plus_Full', 'QI',
       'Other_full', 'Public_Total', 'SNAP_PA_Participation_Persons',
       'SNAP_NPA_Participation_Persons', 'SNAP_All_Participation_Persons',
       'SNAP_PA_Participation_Households', 'SNAP_NPA_Participation_Households',
       'SNAP_All_Participation_Households', 'SNAP_All_Issuance',
       'deaths_per_100k', 'cases_per_100k', 'jobs_per_100k',
       'av_household_earnings_per_100k']]


    # drop Alaska and Hawaii
    df_lower = covid_df[(covid_df['AK'] == 0) & (covid_df['HI'] == 0)]

    df_lower.rename(columns={'pct_black':'pct_African_American'}, inplace=True)
    df_lower.rename(columns={'pct_hispanic':'pct_Latinx'}, inplace=True)
    df_lower.rename(columns={'pct_asian':'pct_Asian'}, inplace=True)
    df_lower.rename(columns={'pct_native_american':'pct_Native_American'}, inplace=True)

    data_set = date + '_cov_soc_eco_lower48.csv'
    # Export to data store
    dated_buffer = StringIO()
    df_lower.to_csv(dated_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, f'{archive_path}/{data_set}').put(Body=dated_buffer.getvalue())

    #commenting out next line as change to StringIO
    #df_lower.to_csv(archive_path + data_set)

    # Export to daily update store
    csv_buffer = StringIO()
    df_lower.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, f'{dashboard_datapath}/daily_update.csv').put(Body=csv_buffer.getvalue())
    
    #commenting out next line as change to StringIO
    #df_lower.to_csv(dashboard_datapath + 'daily_update.csv')


    # print the cases and deaths numbers from the final data_set
    print("Think about changing this to be today's numbers:")
    print('cases total on new data set: ', df_lower['confirmed_cases'].sum())
    print('deaths total on new data set: ', df_lower['deaths'].sum())

    print(f'\nSUCCESS, completed creating {data_set} and exported to {datapath}\n')

    
