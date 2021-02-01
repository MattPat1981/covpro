## sccounty.py runs a local forecast for 
#  Santa Cruz County, California

#  We can package the whole thing and swap out counties when we want


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import boto3
from io import StringIO


try:
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.mx.trainer   import Trainer
    from gluonts.dataset.common import ListDataset
    from gluonts.evaluation.backtest import make_evaluation_predictions
except:
    print('Please pip install --upgrade mxnet==1.6.0')
    print('Please pip install gluonts')

import Steinbeck as sbk
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['axes.grid'] = False
commence_time = str(datetime.datetime.now())

county = 'Santa Cruz County, California' # Forecast for this county
train_time = str(datetime.date.today() - datetime.timedelta(days=1))  # Train the model with data until this day
prediction_length=14                     # Predicting for x days in the future
forecast_time = str(datetime.date.today() + datetime.timedelta(days=prediction_length-1))            # predition until this date on unknkown data

num_layers = 2   # How many layers in the neural network?
num_cells = 16   # How many cells in the neural network layers
num_samples = 200 # How many times do we run the monte carlo proba run

future_dates = []
new_day = datetime.date.today()
for i in range(14):
    future_dates.append(str(new_day))
    new_day = new_day + datetime.timedelta(days=1)


# Set datapath
bucket = 'covid-v1-part-3-data-bucket'
prefix = 'timeseries/data' # change to your desired S3 prefix
#case_file = 'time_series_covid19_confirmed_US.csv'
#death_file = 'time_series_covid19_deaths_US.csv'
#timeseries_data_path = '{}/{}/{}'.format(bucket, prefix, datafile)
datapath = 's3://{}/{}/'.format(bucket, prefix)
county_path = datapath + 'daily/all_county_time_series/'

# Pull in the data
df = sbk.get_county(county, county_path)
# Update the new dataset with the future_dates Months
for date in future_dates:
    df.loc[date] = [0, 0, int(date[5:7]), 0, 0, 0, 0]

# set the index of the dataframe to datetime for timesereis fc
df.index = pd.to_datetime(df.index)

# Prepare the data for modeling
df_input=df[['new_daily_cases', 'new_daily_deaths', 'Month', 'cases_per_100K', 'deaths_per_100K']]

training_data = ListDataset(
    [{"start": df_input.index[0], "target": df_input.new_daily_cases[:train_time]}],
    freq = "D",
)

# Give dictionaries / JSON for the test sets (time series)
# Unclear why the tutorials thus far start at t=0 instaead of the prediction date

test_data = ListDataset(
    [
        {"start": df_input.index[0], "target": df_input.new_daily_cases[:train_time]},
        {"start": df_input.index[0], "target": df_input.new_daily_cases[:forecast_time]}
    ],
    freq = "D"
)

# Create the model object

estimator = DeepAREstimator(freq="D",
                           context_length=14,  # How many past events do I look at to make prediction
                           prediction_length=prediction_length,
                           num_layers=num_layers,
                           num_cells=num_cells,
                           #num_parallel_samples=8, # Added 12/22/2020 -- Doesn't seem to be working in parallel
                           dropout_rate=0.1,      # Added 12/22/2020
                           cell_type='lstm',
                           trainer=Trainer(epochs=21))  # modify as needed


# Train the model on the json version (created in the step above)
# of the training portion of the data set
predictor = estimator.train(training_data=training_data)


# Set up the data results of the model
forecast_it, ts_it = make_evaluation_predictions(
    dataset = test_data,
    predictor = predictor,
    num_samples = num_samples,    # This is running x times through the probablistic model
)
forecasts = list(forecast_it)
tss = list(ts_it)

# This function is taken from a tutorial. Still, with some tweaking and citations, it sould be added to Steinbeck.py

def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = prediction_length
    prediction_intervals = (80.0, 95.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which='both')
    plt.legend(legend, loc="upper left")
    plt.title(county + ' ' + str(prediction_length) + '-day forecast', fontsize=24);
    plt.show()



# # uncomment this block in jupyter notebook or to render the graph


# plot_prob_forecasts(tss[0], forecasts[0])
# plot_prob_forecasts(tss[-1], forecasts[-1])
# # end graph end

filename = county.replace(',','')
print('exporting data for ' + filename)

#Send to csv in s3
csv_buffer = StringIO()
forecasts[-1].mean_ts.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, f"/DeepAR/predictions/{filename.replace(' ','_')}.csv").put(Body=csv_buffer.getvalue())


print('Exported {}-day forecast to {}.csv'.format(prediction_length, filename.replace(' ','_')))

# Next, run the new_daily_deaths forecast, adjusting 
# the settings on DeepAR's Recurrent Neural Network
# QUICK SEARCH to here to change to a moving average model

df_input=df[['new_daily_cases', 'new_daily_deaths', 'Month', 'cases_per_100K', 'deaths_per_100K']]

training_data = ListDataset(
    [{"start": df_input.index[0], "target": df_input.new_daily_deaths[:train_time]}],
    freq = "D",
)

# Give dictionaries / JSON for the test sets (time series)
# Unclear why the tutorials thus far start at t=0 instaead of the prediction date

test_data = ListDataset(
    [
        {"start": df_input.index[0], "target": df_input.new_daily_deaths[:train_time]},
        {"start": df_input.index[0], "target": df_input.new_daily_deaths[:forecast_time]}
    ],
    freq = "D"
)

# Special settings for daily_deaths forecast
num_layers=2
num_cells=16
epochs=2
num_samples = 10
#print(f'num_layers = {num_layers}\nnum_cells = {num_cells}\nepochs = {epochs}\nnum_samples = {num_samples}')

# Create the model object for estimating the new_daily_deaths column

estimator_d = DeepAREstimator(freq="D",
                           context_length=14,  # How many past events do I look at to make prediction
                           prediction_length=prediction_length,
                           num_layers=num_layers,
                           num_cells=num_cells,
                           #num_parallel_samples=8, # Added 12/22/2020 -- Doesn't seem to be working in parallel
                           dropout_rate=0.1,      # Added 12/22/2020
                           cell_type='lstm',
                           trainer=Trainer(epochs=epochs))  # modify as needed

#%time

# Train the model on the json version (created in the step above)
# of the training portion of the data set
predictor = estimator_d.train(training_data=training_data)


forecast_it, ts_it = make_evaluation_predictions(
    dataset = test_data,
    predictor = predictor,
    num_samples = num_samples,    # This is running x times through the probablistic model
)
forecasts = list(forecast_it)
tss = list(ts_it)

# # uncomment this block in jupyter notebook or to render the graph

# plot_prob_forecasts(tss[0], forecasts[0])
# plot_prob_forecasts(tss[-1], forecasts[-1])
# ## End graphing block

#Send to csv in repo
csv_buffer = StringIO()
forecasts[-1].mean_ts.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, f"/DeepAR/predictions/{filename.replace(' ','_')}_deaths.csv").put(Body=csv_buffer.getvalue())

print('Exported {}-day forecast to {}_deaths.csv'.format(prediction_length, filename.replace(' ','_')))

# Next, prepare the two forecasts into one csv to update the graphic
# Incorporate the CA Restrictions Tiers as well

df = sbk.get_county(county, county_path)

res = pd.read_csv(f's3://{bucket}/timeseries/restrictions.csv')

county = 'Santa Cruz County, California'
cname = county.replace(' County, California', '')

res.date = pd.to_datetime(res.date)
res.set_index('date', inplace=True)
sc_res = res[res.county==cname]
big = df.join(sc_res, on='date', how='left')

big.drop(columns='county', inplace=True)

big['tier'].loc['2020-01-22'] = 4
big['tier'].loc['2020-03-16'] = 1
big['tier'].loc['2020-05-04'] = 2
big['tier'].loc['2020-06-12'] = 3 # State allows stores to re-open, two days later
                                  # the state sets a hospitalization record

#Send this to csv
county = county.replace(",","")

#Send to csv in repo
csv_buffer = StringIO()
big.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, f"{prefix}restrictions-folder/{county.replace(' ','_')}.csv").put(Body=csv_buffer.getvalue())


prefix_one = 'timeseries/data/restrictions-folder/' # change to your desired S3 prefix
prefix_two = 'DeepAR/predictions/'
datapath = 's3://{}/'.format(bucket)

scd = pd.read_csv(datapath + prefix_one + county.replace(" ","_") + '.csv')

fcc = pd.read_csv(datapath + prefix_two + filename.replace(" ","_") + '.csv')
fcc.columns = ['date', 'new_daily_cases']


fcd = pd.read_csv(datapath + prefix_two + filename.replace(" ","_") + '_deaths.csv')
fcd.columns = ['date', 'new_daily_deaths']

# REMOVE this block when updating with the moving average model

def round_the_dead(data):
    if data < 0.02:
        return 0
#     if data > 1.1:
#         return np.ceil(data)
    else:
        return np.ceil(data)

fcd.new_daily_deaths = fcd.new_daily_deaths.apply(round_the_dead)

# append the daily deaths data to the end of the full data set
model = pd.concat([scd, fcd])

### Merge the daily cases data to the full data set

model = model.merge(fcc, how='outer', left_on='date', right_on='date')

### Copy over the cases data to the correct column and eliminate the extra column

model.new_daily_cases_x = model.new_daily_cases_x.fillna(model.new_daily_cases_y)

model.drop(columns='new_daily_cases_y', inplace=True)
model.rename(columns=({'new_daily_cases_x':'new_daily_cases'}), inplace=True)

## Set the date column to be the index

model['date'] = pd.to_datetime(model['date'])
model.set_index('date', inplace=True)

model.loc['2021-02-01']['Month'] = 2


model['Month'] = model['Month'].fillna(method='ffill')

### Fill in the cases and deaths numbers with the forecasted changes

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

model.reset_index(level=0, inplace=True)
index = list(model.index)
for i in range(len(index)-14 , len(index)):
    model['cases'].iloc[i] = (model['cases'].iloc[i-1] + model['new_daily_cases'].iloc[i]).copy()
    model['tier'].iloc[i] = 5
    model['deaths'].iloc[i] = (model['deaths'].iloc[i-1] + model['new_daily_deaths'].iloc[i]).copy()


model = model.fillna(method='ffill')

export_path = 'timeseries/dashboard-data/'

#Send to csv in repo
csv_buffer = StringIO()
model.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, f"{prefix}/{export_path}{filename.replace(' ','_')}.csv").put(Body=csv_buffer.getvalue())



print('Exported {}.csv to s3 bucket {}'.format(filename.replace(" ", "_"), datapath+export_path))

program_done = str(datetime.datetime.now())

print('start           : ' + commence_time)
print('models complete : '+ program_done)

