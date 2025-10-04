import os
import torch
import numpy as np

class EEGNet(torch.nn.Module):
    def __init__(self, nb_classes, Chans=40, Samples=2500, dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        if dropoutType == 'SpatialDropout2D':
            self.DropoutType = torch.nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.DropoutType = torch.nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

        self.conv1 = torch.nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(F1)
        self.depthwise_conv = torch.nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False, padding='valid')
        self.batchnorm2 = torch.nn.BatchNorm2d(F1 * D)
        self.elu1 = torch.nn.ELU()
        self.avgpool1 = torch.nn.AvgPool2d((1, 4))
        self.dropout1 = self.DropoutType(dropoutRate)
        
        self.separable_conv = torch.nn.Sequential(
            torch.nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False, padding='same'),
            torch.nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        )
        self.batchnorm3 = torch.nn.BatchNorm2d(F2)
        self.elu2 = torch.nn.ELU()
        self.avgpool2 = torch.nn.AvgPool2d((1, 8))
        self.dropout2 = self.DropoutType(dropoutRate)
        
        self.flatten = torch.nn.Flatten()
        self.dense = torch.nn.Linear(F2 * (Samples // 32), nb_classes)
        
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        return x

def inference(input_data, model_path, class_labels=["Healthy", "Parkinson's Disease"], device='cpu'):

    if len(class_labels) != 2:
        raise ValueError("class_labels must be a list of exactly two strings.")
    
    if isinstance(input_data, list):
        input_data = np.array(input_data, dtype=np.float32)

    elif not isinstance(input_data, np.ndarray):
        raise ValueError("Input data must be a numpy array or list.")
    
    if len(input_data.shape) != 3 or input_data.shape[1] != 40 or input_data.shape[2] != 2500:
        raise ValueError(f"Input data must have shape (batch_size, 40, 2500). Got: {input_data.shape}")
    

    model = EEGNet(2, 40, 2500,  0.5, 32, 8, 2, 2*8, 0.25, 'Dropout')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # Convert input to torch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().tolist()  
        predictions = torch.argmax(outputs, dim=1).cpu().numpy().tolist()  
        labeled_predictions = [class_labels[p] for p in predictions]
    
    return {
        'labeled_predictions': labeled_predictions,
        'probabilities': probabilities
    }


# if __name__ == "__main__":
#     import random
#     import pandas as pd
#     from mne_bids import BIDSPath, read_raw_bids
#     import mne

#     dataset_dir = 'ds002778-download'

#     # Load participants.tsv
#     participants_path = os.path.join(dataset_dir, 'participants.tsv')
#     df = pd.read_csv(participants_path, sep='\t')

#     # Pick a random subject
#     idx = random.randint(0, len(df) - 1)
#     subject = df['participant_id'][idx]
#     true_label = "Healthy" if subject.startswith('sub-hc') else "Parkinson's Disease"

#     print(f"Selected subject: {subject}, True label: {true_label}")

#     # Find available sessions for the subject
#     subject_dir = os.path.join(dataset_dir, subject)
#     ses_dirs = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d)) and d.startswith('ses-')]
#     if ses_dirs:
#         ses = random.choice(ses_dirs)[4:]
#     else:
#         ses = None

#     print(f"Selected session: {ses}")

#     # Load the EEG data using MNE-BIDS
#     bids_root = dataset_dir
#     bids_path = BIDSPath(subject=subject[4:], session=ses, task='rest', datatype='eeg', root=bids_root)
#     raw = read_raw_bids(bids_path, verbose=False)
#     raw.load_data()

#     # Preprocessing
#     raw.filter(l_freq=0.5, h_freq=None)  # High-pass filter
#     raw.resample(500)  # Resample to 500 Hz

#     # Select EEG channels
#     raw.pick_types(eeg=True)

#     # Get data
#     data = raw.get_data()

#     if data.shape[0] != 40:
#         print(f"Warning: Number of channels is {data.shape[0]}, expected 40.")

#     # Pick a random 5-second segment (2500 samples at 500 Hz)
#     sfreq = raw.info['sfreq']
#     segment_length = 2500
#     start_sample = random.randint(0, data.shape[1] - segment_length)
#     segment = data[:, start_sample:start_sample + segment_length]

#     # Prepare input for model: (1, channels, time)
#     input_data = segment[np.newaxis, :, :].astype(np.float32)

#     # Run inference
#     model_path = 'models/eegnet_finetuned.pt'  # Replace with actual path
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     result = inference(input_data, model_path, device=device)
#     print("Predicted label:", result['labeled_predictions'][0])
#     print("Probability for Parkinson's Disease:", result['probabilities'][0])