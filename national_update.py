import Steinbeck as sbk

datapath    = '/home/ubuntu/covpro/data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/' 
output_path = 'timeseries/data/daily/all_county_time_series'

def main():
    # Run the national_update() function to create 
    # dataframes for each county and save them 
    # in folder, overwriting the old version daily
    sbk.national_update(datapath, output_path)
    
    print('national_update sent to s3://{covid-v1-part-3-data-bucket}/{output_path}/')

if __name__ == "__main__":
    main()

