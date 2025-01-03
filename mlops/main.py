from utils.fake_account_detector import FakeAccountDetector
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
connection_string = os.getenv("MONGO_CONN_STRING")
client = MongoClient(connection_string)
db = client['fake_account_data'] 

real_users_collection = db['real_users']
fake_users_collection = db['fake_users']
real_users = pd.DataFrame(list(real_users_collection.find()))
fake_users = pd.DataFrame(list(fake_users_collection.find()))

        
if "_id" in real_users.columns:
    data = real_users.drop(columns=["_id"])
data.to_csv('C:/Users/Aravind/fake-data-detection/mlops/data/users.csv', index=False)

if "_id" in fake_users.columns:
    data = fake_users.drop(columns=["_id"])
data.to_csv('C:/Users/Aravind/fake-data-detection/mlops/data/fake_users.csv', index=False)


detector = FakeAccountDetector()


X, y = detector.load_and_preprocess_data("C:/Users/Aravind/fake-data-detection/mlops/data/users.csv", "C:/Users/Aravind/fake-data-detection/mlops/data/fake_users.csv")
results = detector.train(X, y)
    

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

detector.plot_feature_importance()
detector.save_models("C:/Users/Aravind/fake-data-detection/mlops/models")