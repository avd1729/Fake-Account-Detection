from utils.fake_account_detector import FakeAccountDetector

detector = FakeAccountDetector()
    
X, y = detector.load_and_preprocess_data("C:/Users/Aravind/fake-data-detection/mlops/data/users.csv", "C:/Users/Aravind/fake-data-detection/mlops/data/fake_users.csv")
results = detector.train(X, y)
    

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

detector.plot_feature_importance()
detector.save_models("C:/Users/Aravind/fake-data-detection/mlops/models")