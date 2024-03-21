import pandas as pd
import os
from constants import YEAR, DATA_DIR

months_order = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

def combine_schedules():
    # Define the order of the months

    # List of dataframes to store each month's data
    dfs = []

    # Iterate over the months in the specified order
    for month in months_order:
        # Construct the file name based on the month
        file_name = f'{DATA_DIR}/{YEAR}_Schedule_{month}.csv'
        
        # Check if the file exists
        if os.path.exists(file_name):
            # Read the CSV file and append to the list of dataframes
            df = pd.read_csv(file_name)
            dfs.append(df)
        else:
            print(f'File {file_name} does not exist.')

    # Concatenate all dataframes into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(f'{DATA_DIR}/2023_combined_schedule.csv', index=False)

    # Print a success message
    print('All CSV files have been successfully combined into combined_schedule.csv.')


def remove_old_schedules():
    for month in months_order:
        file_name = f'{DATA_DIR}/{YEAR}_Schedule_{month}.csv'
        
        # Check if the file exists
        if os.path.exists(file_name):
            os.remove(file_name)
        else:
            print(f'File {file_name} does not exist.')
            
    print('Removed all old schedule files.')


combine_schedules()
remove_old_schedules()