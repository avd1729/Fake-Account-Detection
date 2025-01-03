import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

connection_string = os.getenv("MONGO_CONN_STRING")
client = MongoClient(connection_string)
db = client['fake_account_data'] 

real_users_df = None
fake_users_df = None

def extract_data_from_csv():

    global real_users_df, fake_users_df
    load_dotenv()

    real_users = 'data/users.csv' 
    fake_users = 'data/fake_users.csv'

    real_users_df = pd.read_csv(real_users)
    fake_users_df = pd.read_csv(fake_users)

def transform_data():
    pass

def load_data_to_mongodb():

    real_users_collection = db['real_users']
    fake_users_collection = db['fake_users']
    
    real_users_collection.drop()
    fake_users_collection.drop()

    real_users_collection.insert_many(real_users_df.to_dict('records'))
    fake_users_collection.insert_many(fake_users_df.to_dict('records'))

    print("Data loaded successfully!")

