

import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from mne.io import read_raw_edf
import warnings
warnings.filterwarnings('ignore')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass


def predict_seizure(edf_path, model_path='epilepsy_eegnet_model.h5', scaler_path='epilepsy_scaler.joblib'):

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    raw = read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    
    window_samples = 256  
    max_channels = 23
    stride = 256  
    batch_size = 128  
    
    if data.shape[0] > max_channels:
        data = data[:max_channels, :]
    elif data.shape[0] < max_channels:
        padding = np.zeros((max_channels - data.shape[0], data.shape[1]))
        data = np.vstack([data, padding])
    
    n_samples = data.shape[1]
    windows = []
    
    for start_idx in range(0, n_samples - window_samples, stride):
        end_idx = start_idx + window_samples
        window = data[:, start_idx:end_idx]
        windows.append(window)
    
    if len(windows) == 0:
        return 0
    
    windows = np.array(windows)  
    
    windows_flat = windows.reshape(len(windows), -1)
    windows_normalized = scaler.transform(windows_flat).reshape(len(windows), max_channels, window_samples, 1)
    
    predictions = model.predict(windows_normalized, batch_size=batch_size, verbose=0)
    
    seizure_count = np.sum(predictions > 0.5)
    total_count = len(predictions)
    
    seizure_ratio = seizure_count / total_count if total_count > 0 else 0
    return "seizure" if seizure_ratio > 0.1 else "no seizure"
