import tensorflow as tf
import joblib
import mne
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class Config:
    """Centralized configuration class for all parameters."""

    # Data Configuration
    SAMPLING_RATE = 160 # Hz
    N_CHANNELS = 64 # Number of EEG channels
    N_CLASSES = 2 # Left vs. Right hand imagery

    # Subject Configuration (Robust Splitting)
    # Total subjects: 109. We use a subject-wise split for robust validation.
    ALL_SUBJECTS = list(range(1, 110))
    TRAIN_SUBJECTS = list(range(1, 110)) # Subjects 1-80 for training (approx. 75%)
    VALID_SUBJECTS = list(range(51, 110)) # Subjects 81-95 for validation (approx. 15%)
    TEST_SUBJECTS = list(range(51, 110)) # Subjects 96-109 for final testing (approx. 15%)

    # Signal Processing Parameters
    LOWCUT = 8.0 # Lower cutoff frequency (Hz) for Mu rhythm
    HIGHCUT = 30.0 # Upper cutoff frequency (Hz) for Beta rhythm
    FILTER_ORDER = 5 # Butterworth filter order
    WINDOW_SIZE = 4.0 # Time window in seconds (full imagery period)
    T_MIN = 0.0 # Start time of the epoch relative to the event
    T_MAX = 4.0 # End time of the epoch

    # EEGNet Architecture Parameters
    F1 = 8 # Number of temporal filters
    D = 2 # Number of spatial filters
    F2 = 16 # Number of pointwise filters
    KERNEL_LENGTH = 64 # Length of the temporal convolution kernel
    DROPOUT_RATE = 0.5 # Dropout rate for regularization

    # Training Parameters
    BATCH_SIZE = 16 # Smaller batch size for better generalization on EEG data
    EPOCHS = 200 # Max epochs, with early stopping
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 30 # Increased patience for EEGNet

    # Visualization with Plotly
    PLOTLY_TEMPLATE = 'plotly_dark'

config = Config()


def predict_on_edf(file_path, model, scaler, config_params):
    """
    Loads a single .edf file, preprocesses it, and returns model predictions
    along with the true labels and processed data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load and preprocess the raw data
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.filter(config_params.LOWCUT, config_params.HIGHCUT, fir_design='firwin', skip_by_annotation='edge', verbose=False)

    # 2. Extract events and create epochs
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if not any(desc in ['T1', 'T2'] for desc in raw.annotations.description):
        print("No 'T1' or 'T2' annotations found in the file. Cannot create epochs.")
        return None, None, None

    event_mapping = {'T1': 2, 'T2': 3}

    # FIX: Explicitly set baseline=None to disable baseline correction and avoid errors.
    epochs = mne.Epochs(raw, events, event_id=event_mapping, tmin=config_params.T_MIN,
                        tmax=config_params.T_MAX, preload=True, verbose=False, baseline=None)

    eeg_data = epochs.get_data(copy=False)
    # Extract true labels from the epoch events
    true_labels = epochs.events[:, -1] - 2  # Map back to 0 and 1

    # 3. Scale and reshape the data
    eeg_data_flat = eeg_data.reshape(eeg_data.shape[0], -1)
    eeg_data_scaled_flat = scaler.transform(eeg_data_flat)
    eeg_data_scaled = eeg_data_scaled_flat.reshape(eeg_data.shape)
    eeg_data_final = eeg_data_scaled[:, :, :, np.newaxis]

    # 4. Make predictions
    predictions_prob = model.predict(eeg_data_final, batch_size=config_params.BATCH_SIZE, verbose=0)
    predictions_class = np.argmax(predictions_prob, axis=1)

    return predictions_class, true_labels, eeg_data_final

def visualize_prediction_grid(processed_data, predictions, true_labels, config_params):
    """Visualizes all trials in a grid with ground truth, predictions, and performance metrics."""
    class_labels = {0: 'Left Hand', 1: 'Right Hand'}
    n_trials = processed_data.shape[0]

    # Determine grid size (e.g., 4 columns)
    n_cols = 4
    n_rows = (n_trials + n_cols - 1) // n_cols

    # Calculate overall accuracy and other metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=class_labels.values(), output_dict=True)

    # Create the main title with metrics
    main_title = (
        f"<b>Motor Imagery Classification Results for EDF File</b><br>"
        f"Overall Accuracy: <span style='color: #17BECF;'>{accuracy*100:.2f}%</span>"
    )

    # Create subplot grid
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Trial {i+1}" for i in range(n_trials)],
        vertical_spacing=0.08,
        horizontal_spacing=0.04
    )

    time_axis = np.linspace(config_params.T_MIN, config_params.T_MAX, processed_data.shape[2])
    channels_to_plot = {'C3': 10, 'Cz': 12, 'C4': 14}
    colors = ['#636EFA', '#EF553B', '#00CC96']

    for i in range(n_trials):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Plot EEG signals for the key channels
        for j, (ch_name, ch_idx) in enumerate(channels_to_plot.items()):
            fig.add_trace(go.Scatter(
                x=time_axis, y=processed_data[i, ch_idx, :, 0],
                mode='lines', name=ch_name, legendgroup=ch_name,
                showlegend=(i == 0), line=dict(color=colors[j])
            ), row=row, col=col)

        # Determine color for the prediction text (green for correct, red for incorrect)
        pred_label = class_labels.get(predictions[i])
        gt_label = class_labels.get(true_labels[i])
        color = '#00CC96' if predictions[i] == true_labels[i] else '#EF553B'

        # Update subplot title with Ground Truth and Prediction
        new_title = f"Trial {i+1}<br>GT: {gt_label} | <span style='color:{color};'>Pred: {pred_label}</span>"
        fig.layout.annotations[i].update(text=new_title)

    fig.update_layout(
        height=250 * n_rows, title_text=main_title,
        template=config_params.PLOTLY_TEMPLATE, showlegend=True,
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_font=dict(size=10))
    fig.update_yaxes(title_font=dict(size=10))
    fig.show()

    # Print the detailed classification report
    print("📋 Classification Report for this file:")
    print(classification_report(true_labels, predictions, target_names=class_labels.values()))
 
# EXAMPLE CODE FROM NOTEBOOK
print(" Launching Inference Pipeline Example...")
try:
    # 1. Load the trained model and scaler
    loaded_model = tf.keras.models.load_model('best_eegnet_model.keras')
    loaded_scaler = joblib.load('eeg_scaler.gz')
    print(" Model and scaler loaded successfully.")

    # 2. Set the path to the user's .edf file
    EXAMPLE_EDF_PATH = '/kaggle/input/eeg-motor-movementimagery-dataset/files/S002/S002R03.edf'
    print(f" Attempting to predict on: {EXAMPLE_EDF_PATH}")

    # 3. Get predictions and true labels
    predictions, true_labels, processed_data = predict_on_edf(EXAMPLE_EDF_PATH, loaded_model, loaded_scaler, config)

    if predictions is not None:
        # 4. Visualize all trials in a grid
        print("\n Visualizing all trial predictions for the file...")
        visualize_prediction_grid(processed_data, predictions, true_labels, config)

except FileNotFoundError as e:
    print(f"\n ERROR: {e}")
    print("Please ensure the file path is correct and the dataset is available.")
except Exception as e:
    print(f"\n An unexpected error occurred: {e}")
import tensorflow as tf
import joblib
import mne
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class Config:
    """Centralized configuration class for all parameters."""

    # Data Configuration
    SAMPLING_RATE = 160 # Hz
    N_CHANNELS = 64 # Number of EEG channels
    N_CLASSES = 2 # Left vs. Right hand imagery

    # Subject Configuration (Robust Splitting)
    # Total subjects: 109. We use a subject-wise split for robust validation.
    ALL_SUBJECTS = list(range(1, 110))
    TRAIN_SUBJECTS = list(range(1, 110)) # Subjects 1-80 for training (approx. 75%)
    VALID_SUBJECTS = list(range(51, 110)) # Subjects 81-95 for validation (approx. 15%)
    TEST_SUBJECTS = list(range(51, 110)) # Subjects 96-109 for final testing (approx. 15%)

    # Signal Processing Parameters
    LOWCUT = 8.0 # Lower cutoff frequency (Hz) for Mu rhythm
    HIGHCUT = 30.0 # Upper cutoff frequency (Hz) for Beta rhythm
    FILTER_ORDER = 5 # Butterworth filter order
    WINDOW_SIZE = 4.0 # Time window in seconds (full imagery period)
    T_MIN = 0.0 # Start time of the epoch relative to the event
    T_MAX = 4.0 # End time of the epoch

    # EEGNet Architecture Parameters
    F1 = 8 # Number of temporal filters
    D = 2 # Number of spatial filters
    F2 = 16 # Number of pointwise filters
    KERNEL_LENGTH = 64 # Length of the temporal convolution kernel
    DROPOUT_RATE = 0.5 # Dropout rate for regularization

    # Training Parameters
    BATCH_SIZE = 16 # Smaller batch size for better generalization on EEG data
    EPOCHS = 200 # Max epochs, with early stopping
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 30 # Increased patience for EEGNet

    # Visualization with Plotly
    PLOTLY_TEMPLATE = 'plotly_dark'

config = Config()


def predict_on_edf(file_path, model, scaler, config_params):
    """
    Loads a single .edf file, preprocesses it, and returns model predictions
    along with the true labels and processed data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load and preprocess the raw data
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.filter(config_params.LOWCUT, config_params.HIGHCUT, fir_design='firwin', skip_by_annotation='edge', verbose=False)

    # 2. Extract events and create epochs
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if not any(desc in ['T1', 'T2'] for desc in raw.annotations.description):
        print("No 'T1' or 'T2' annotations found in the file. Cannot create epochs.")
        return None, None, None

    event_mapping = {'T1': 2, 'T2': 3}

    # FIX: Explicitly set baseline=None to disable baseline correction and avoid errors.
    epochs = mne.Epochs(raw, events, event_id=event_mapping, tmin=config_params.T_MIN,
                        tmax=config_params.T_MAX, preload=True, verbose=False, baseline=None)

    eeg_data = epochs.get_data(copy=False)
    # Extract true labels from the epoch events
    true_labels = epochs.events[:, -1] - 2  # Map back to 0 and 1

    # 3. Scale and reshape the data
    eeg_data_flat = eeg_data.reshape(eeg_data.shape[0], -1)
    eeg_data_scaled_flat = scaler.transform(eeg_data_flat)
    eeg_data_scaled = eeg_data_scaled_flat.reshape(eeg_data.shape)
    eeg_data_final = eeg_data_scaled[:, :, :, np.newaxis]

    # 4. Make predictions
    predictions_prob = model.predict(eeg_data_final, batch_size=config_params.BATCH_SIZE, verbose=0)
    predictions_class = np.argmax(predictions_prob, axis=1)

    return predictions_class, true_labels, eeg_data_final

def visualize_prediction_grid(processed_data, predictions, true_labels, config_params):
    """Visualizes all trials in a grid with ground truth, predictions, and performance metrics."""
    class_labels = {0: 'Left Hand', 1: 'Right Hand'}
    n_trials = processed_data.shape[0]

    # Determine grid size (e.g., 4 columns)
    n_cols = 4
    n_rows = (n_trials + n_cols - 1) // n_cols

    # Calculate overall accuracy and other metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=class_labels.values(), output_dict=True)

    # Create the main title with metrics
    main_title = (
        f"<b>Motor Imagery Classification Results for EDF File</b><br>"
        f"Overall Accuracy: <span style='color: #17BECF;'>{accuracy*100:.2f}%</span>"
    )

    # Create subplot grid
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Trial {i+1}" for i in range(n_trials)],
        vertical_spacing=0.08,
        horizontal_spacing=0.04
    )

    time_axis = np.linspace(config_params.T_MIN, config_params.T_MAX, processed_data.shape[2])
    channels_to_plot = {'C3': 10, 'Cz': 12, 'C4': 14}
    colors = ['#636EFA', '#EF553B', '#00CC96']

    for i in range(n_trials):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Plot EEG signals for the key channels
        for j, (ch_name, ch_idx) in enumerate(channels_to_plot.items()):
            fig.add_trace(go.Scatter(
                x=time_axis, y=processed_data[i, ch_idx, :, 0],
                mode='lines', name=ch_name, legendgroup=ch_name,
                showlegend=(i == 0), line=dict(color=colors[j])
            ), row=row, col=col)

        # Determine color for the prediction text (green for correct, red for incorrect)
        pred_label = class_labels.get(predictions[i])
        gt_label = class_labels.get(true_labels[i])
        color = '#00CC96' if predictions[i] == true_labels[i] else '#EF553B'

        # Update subplot title with Ground Truth and Prediction
        new_title = f"Trial {i+1}<br>GT: {gt_label} | <span style='color:{color};'>Pred: {pred_label}</span>"
        fig.layout.annotations[i].update(text=new_title)

    fig.update_layout(
        height=250 * n_rows, title_text=main_title,
        template=config_params.PLOTLY_TEMPLATE, showlegend=True,
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_font=dict(size=10))
    fig.update_yaxes(title_font=dict(size=10))
    fig.show()

    # Print the detailed classification report
    print("📋 Classification Report for this file:")
    print(classification_report(true_labels, predictions, target_names=class_labels.values()))
 
# EXAMPLE CODE FROM NOTEBOOK
print(" Launching Inference Pipeline Example...")
try:
    # 1. Load the trained model and scaler
    loaded_model = tf.keras.models.load_model('best_eegnet_model.keras')
    loaded_scaler = joblib.load('eeg_scaler.gz')
    print(" Model and scaler loaded successfully.")

    # 2. Set the path to the user's .edf file
    EXAMPLE_EDF_PATH = '/kaggle/input/eeg-motor-movementimagery-dataset/files/S002/S002R03.edf'
    print(f" Attempting to predict on: {EXAMPLE_EDF_PATH}")

    # 3. Get predictions and true labels
    predictions, true_labels, processed_data = predict_on_edf(EXAMPLE_EDF_PATH, loaded_model, loaded_scaler, config)

    if predictions is not None:
        # 4. Visualize all trials in a grid
        print("\n Visualizing all trial predictions for the file...")
        visualize_prediction_grid(processed_data, predictions, true_labels, config)

except FileNotFoundError as e:
    print(f"\n ERROR: {e}")
    print("Please ensure the file path is correct and the dataset is available.")
except Exception as e:
    print(f"\n An unexpected error occurred: {e}")