import joblib
import mne
import numpy as np
import antropy as ant
import os
import warnings

FREQ_BANDS = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha": [8, 13],
    "beta": [13, 30]
}
mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

def predict(file: str, model: str = "alzheimer.joblib") :
 
    try:
        # loading model 
        model = joblib.load(model)
        raw = mne.io.read_raw_eeglab(file, preload=True)
        
        # preprocessing
        raw.pick_types(eeg=True)
        raw.filter(l_freq=0.5, h_freq=45.0)
        raw.notch_filter(freqs=50.0)
        epochs = mne.make_fixed_length_epochs(raw, duration=4, preload=True)

        if len(epochs) == 0:
            return {"error": " The recording is to too short."}
        
        epochs_data = epochs.get_data()
        sfreq = raw.info['sfreq']

        # features 
        n_epochs, n_channels, _ = epochs_data.shape
        info = mne.create_info(ch_names=n_channels, sfreq=sfreq, ch_types='eeg')
        epochs_obj = mne.EpochsArray(epochs_data, info)
        psds, freqs = epochs_obj.compute_psd(method='welch', fmin=0.5, fmax=45.0).get_data(return_freqs=True)
        
        band_powers = [psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=2) for fmin, fmax in FREQ_BANDS.values()]
        power_features = np.concatenate(band_powers, axis=1)

        samp_entropy_features = np.array([[ant.sample_entropy(ch) for ch in ep] for ep in epochs_data])
        hjorth_activity = np.var(epochs_data, axis=2)
        
        dx = np.diff(epochs_data, axis=2)
        dx_var = np.var(dx, axis=2)
        hjorth_mobility = np.sqrt(dx_var / (hjorth_activity + 1e-7))
        
        ddx = np.diff(dx, axis=2)
        ddx_var = np.var(ddx, axis=2)
        hjorth_complexity = np.sqrt(ddx_var / (dx_var + 1e-7)) / (hjorth_mobility + 1e-7)
        
        features = np.concatenate([
            power_features, samp_entropy_features, hjorth_activity,
            hjorth_mobility, hjorth_complexity
        ], axis=1)

        probabilities = model.predict_proba(features)
        
        ad_confidence = np.mean(probabilities[:, 1])
        diagnosis_label = "Alzheimer's Disease" if ad_confidence > 0.5 else "Healthy Control"

        return {
            "status": "success",
            "diagnosis": diagnosis_label,
            "confidence_score": float(f"{ad_confidence:.4f}"),
            "metadata": {
                "file_processed": os.path.basename(file),
                "total_epochs_analyzed": len(epochs),
                "model_used": os.path.basename(model)
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}