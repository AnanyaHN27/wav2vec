import os
import librosa
import soundfile as sf
from pathlib import Path

def load_audio_files(folder_path):
    audio_data = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".flac"):
                file_path = os.path.join(root, file)
                audio, _ = librosa.load(file_path, sr=None)
                audio_data.append(audio)

    return audio_data

def save_training_data(folder_path, audio_data, sample_rate=16000, duration=1.0):
    training_folder = os.path.join(folder_path, 'training')
    Path(training_folder).mkdir(parents=True, exist_ok=True)

    for idx, audio in enumerate(audio_data):
        num_samples = int(sample_rate * duration)
        if len(audio) >= num_samples:
            save_path = os.path.join(training_folder, f'sample_{idx}.flac')
            sf.write(save_path, audio[:num_samples], sample_rate)

folder_path = 'LibriSpeech/dev-clean'
audio_data = load_audio_files(folder_path)

save_training_data('LibriSpeech', audio_data)
