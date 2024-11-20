import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml_model import NeuroMLModel

def load_data(data_path):
    """Load and prepare the dataset"""
    df = pd.read_csv(data_path, sep='\t')
    
    # Extract features and labels
    features = []
    labels = []
    
    for index, row in df.iterrows():
        # For each measurement (1-25)
        for i in range(1, 26):
            feature_vector = []
            for feature in ['air_time', 'disp_index', 'gmrt_in_air', 'gmrt_on_paper',
                          'max_x_extension', 'max_y_extension', 'mean_acc_in_air',
                          'mean_acc_on_paper', 'mean_gmrt', 'mean_jerk_in_air',
                          'mean_jerk_on_paper', 'mean_speed_in_air', 'mean_speed_on_paper',
                          'num_of_pendown', 'paper_time', 'pressure_mean', 'pressure_var',
                          'total_time']:
                feature_vector.append(row[f'{feature}{i}'])
            
            features.append(feature_vector)
            labels.append(1 if row['class'] == 'P' else 0)  # Binary classification for now
    
    return np.array(features), np.array(labels)

def main():
    # Load data
    data_path = os.path.join('data', 'raw', 'drawing_data.txt')
    X, y = load_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    model = NeuroMLModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print("Training Set Performance:")
    print(classification_report(y_train, train_pred))
    print("\nTest Set Performance:")
    print(classification_report(y_test, test_pred))
    
    # Save model
    model_path = os.path.join('app', 'static', 'models', 'trained_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == '__main__':
    main()
