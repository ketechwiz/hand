import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class NeuroMLModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'air_time', 'disp_index', 'gmrt_in_air', 'gmrt_on_paper',
            'max_x_extension', 'max_y_extension', 'mean_acc_in_air',
            'mean_acc_on_paper', 'mean_gmrt', 'mean_jerk_in_air',
            'mean_jerk_on_paper', 'mean_speed_in_air', 'mean_speed_on_paper',
            'num_of_pendown', 'paper_time', 'pressure_mean', 'pressure_var',
            'total_time'
        ]

    def extract_features(self, drawing_data):
        """
        Extract features from raw drawing data
        
        Parameters:
        drawing_data (dict): Raw drawing data containing coordinates and timing
        
        Returns:
        numpy.array: Feature vector
        """
        features = {}
        
        # Calculate time-based features
        timestamps = [point['time'] for point in drawing_data]
        features['total_time'] = timestamps[-1] - timestamps[0]
        
        # Calculate air time and paper time
        pen_states = [point.get('penDown', True) for point in drawing_data]
        features['air_time'] = sum(1 for state in pen_states if not state)
        features['paper_time'] = sum(1 for state in pen_states if state)
        
        # Calculate spatial features
        x_coords = [point['x'] for point in drawing_data]
        y_coords = [point['y'] for point in drawing_data]
        features['max_x_extension'] = max(x_coords) - min(x_coords)
        features['max_y_extension'] = max(y_coords) - min(y_coords)
        
        # Calculate speed and acceleration
        speeds = []
        accelerations = []
        jerks = []
        for i in range(1, len(drawing_data)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = x_coords[i] - x_coords[i-1]
                dy = y_coords[i] - y_coords[i-1]
                speed = np.sqrt(dx*dx + dy*dy) / dt
                speeds.append(speed)
                
                if len(speeds) > 1:
                    acceleration = (speeds[-1] - speeds[-2]) / dt
                    accelerations.append(acceleration)
                    
                    if len(accelerations) > 1:
                        jerk = (accelerations[-1] - accelerations[-2]) / dt
                        jerks.append(jerk)
        
        features['mean_speed_in_air'] = np.mean([s for s, down in zip(speeds, pen_states[1:]) if not down])
        features['mean_speed_on_paper'] = np.mean([s for s, down in zip(speeds, pen_states[1:]) if down])
        features['mean_acc_in_air'] = np.mean([a for a, down in zip(accelerations, pen_states[2:]) if not down])
        features['mean_acc_on_paper'] = np.mean([a for a, down in zip(accelerations, pen_states[2:]) if down])
        features['mean_jerk_in_air'] = np.mean([j for j, down in zip(jerks, pen_states[3:]) if not down])
        features['mean_jerk_on_paper'] = np.mean([j for j, down in zip(jerks, pen_states[3:]) if down])
        
        # Calculate pressure-related features (if available)
        pressures = [point.get('pressure', 1.0) for point in drawing_data]
        features['pressure_mean'] = np.mean(pressures)
        features['pressure_var'] = np.var(pressures)
        
        # Calculate displacement index
        total_displacement = sum(np.sqrt((x2-x1)**2 + (y2-y1)**2) 
                               for (x1, y1), (x2, y2) in zip(zip(x_coords[:-1], y_coords[:-1]),
                                                           zip(x_coords[1:], y_coords[1:])))
        direct_displacement = np.sqrt((x_coords[-1]-x_coords[0])**2 + 
                                    (y_coords[-1]-y_coords[0])**2)
        features['disp_index'] = direct_displacement / total_displacement if total_displacement > 0 else 0
        
        # Return features in correct order
        return np.array([features.get(name, 0) for name in self.feature_names])

    def train(self, X, y):
        """Train the model with given features and labels"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
    
    def predict(self, features):
        """Make prediction for given features"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict_proba(features_scaled)[0]
    
    def save_model(self, path):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
