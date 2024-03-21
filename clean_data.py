import pandas as pd
import os
from constants import DATA_DIR


def convert_xls_to_csv(file_path):
    # Read the HTML tables
    tables = pd.read_html(file_path)
    
    # This code assumes that your data is in the first table
    df = tables[0]
    
    # Convert to CSV
    csv_file = file_path.replace('.xls', '.csv')
    df.to_csv(csv_file, index=False, header=True)
   
    # Rename columns that contain 'Unnamed'
    # df.columns = [col if not col.startswith('Unnamed') else '' for col in df.columns]
    
    # Delete the original .xls file
    os.remove(file_path)
    print(f"Deleted original file: {file_path}")


def clean_headers(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Rename columns that contain 'Unnamed'
    df.columns = [col if 'Unnamed' not in col else '' for col in df.columns]
    
    # Save the cleaned data to CSV
    df.to_csv(file_path, index=False, header=True)
    
    print(f"Cleaned file: {file_path}")


def convert_xls():
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.xls'):
            file_path = os.path.join(DATA_DIR, filename)
            convert_xls_to_csv(file_path)
            
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv'):
            file_path = os.path.join(DATA_DIR, filename)
            clean_headers(file_path)

convert_xls()


# def convert_html_table_to_csv(file_path):
#     # Read the HTML tables
#     tables = pd.read_html(file_path)
    
#     # Assume that your data is in the first table
#     df = tables[0]
    
#     # Rename columns that contain 'Unnamed'
#     df.columns = [col if not col.startswith('Unnamed') else '' for col in df.columns]
    
#     # If you want to remove these columns instead, you can use the following line
#     # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
#     # Convert to CSV, removing index and empty header columns
#     csv_file = file_path.replace('.xls', '.csv')
#     df.to_csv(csv_file, index=False, header=True)
