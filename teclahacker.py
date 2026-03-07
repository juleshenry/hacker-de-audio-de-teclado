import os
import glob
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Model architecture (simplified CoAtNet/ResNet style for Mel-Spectrograms)
class KeystrokeCNN(nn.Module):
    def __init__(self, num_classes=36): # A-Z, 0-9
        super(KeystrokeCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def extract_mel_spectrogram(audio_path, sr=44100):
    y, sr = librosa.load(audio_path, sr=sr)
    # Get keystroke onset
    onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    
    spectrograms = []
    # Extract 300ms chunks around each onset
    chunk_samples = int(0.3 * sr)
    for onset in onsets:
        start = librosa.frames_to_samples(onset)
        end = start + chunk_samples
        if end > len(y): continue
        chunk = y[start:end]
        
        melspec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=64)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        
        # Normalize
        melspec_db = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-6)
        spectrograms.append(melspec_db)
        
    return spectrograms

class KeystrokeDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir): continue
            for audio_file in glob.glob(os.path.join(cls_dir, '*.wav')):
                specs = extract_mel_spectrogram(audio_file)
                for spec in specs:
                    self.samples.append(spec)
                    self.labels.append(idx)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # Add channel dimension
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def train_model(data_dir, epochs=10):
    dataset = KeystrokeDataset(data_dir)
    if len(dataset) == 0:
        print("No training data found!")
        return None
        
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = KeystrokeCNN(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), 'keystroke_model.pth')
    
    # Save classes
    with open('classes.txt', 'w') as f:
        f.write('\n'.join(dataset.classes))
        
    return model

def predict_audio(audio_path, model_path='keystroke_model.pth', classes_path='classes.txt'):
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print("Model or classes file not found. Please train first.")
        return
        
    with open(classes_path, 'r') as f:
        classes = f.read().splitlines()
        
    model = KeystrokeCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    specs = extract_mel_spectrogram(audio_path)
    if not specs:
        print("No keystrokes detected in audio.")
        return
        
    predictions = []
    with torch.no_grad():
        for spec in specs:
            x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(classes[predicted.item()])
            
    text = "".join(predictions)
    print(f"Predicted Keystrokes: {text}")
    return text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Acoustic Keystroke Attack (teclahacker)")
    parser.add_argument('--train', type=str, help='Directory containing training audio files organized by class (e.g., data/a/, data/b/, ...)')
    parser.add_argument('--predict', type=str, help='Audio file to predict keystrokes from')
    
    args = parser.parse_args()
    
    if args.train:
        print(f"Training model on data from {args.train}...")
        train_model(args.train)
    elif args.predict:
        print(f"Predicting keystrokes for {args.predict}...")
        predict_audio(args.predict)
    else:
        print("Please specify --train <data_dir> or --predict <audio_file>")
        print("Example 1: python teclahacker.py --train my_keyboard_data/")
        print("Example 2: python teclahacker.py --predict o_zorro_e_gris.wav")
