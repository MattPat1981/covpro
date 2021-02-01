import datetime
today=datetime.datetime.now()
print("Welcome to CovPro, ruuning ETL for COVID-19 dashboard")
print('*'*30)
print(f'Starting CovPro for {str(today)}\n')

from Updates3 import report_date, jhu_date, daily_snapshot, run_regression_models, prep_final_data


# set datapath
bucket = 'covid-v1-part-3-data-bucket'
datapath = 'data'
date=report_date()
covpath = '../data/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'

print('\nStarting the "daily_snapshot" function')
daily_snapshot(datapath, covpath, date )
print('...daily_snapshot completed.\n\nStarting "run_regression_models"')
run_regression_models(datapath, 'updated_snapshot.csv', date=date)
print('...run_regression_models completed.\n\nStarting "prep_final_data"')
prep_final_data(datapath='data', dashboard_datapath='data', archive_path='data/archive', filename='85_cols.csv', date=date)
print('Completed prep_final_data\n\nEOF\n')

