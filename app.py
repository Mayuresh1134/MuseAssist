from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model for audio genre classification
MODEL_PATH = 'model1.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define the genre labels (assuming 10 genres from GTZAN dataset)
GENRE_LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocess the uploaded audio file by extracting MFCCs for classification
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, duration=30)  # Load 30 seconds of audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCC coefficients
    mfccs_scaled = np.mean(mfccs.T, axis=0)  # Mean normalization
    mfccs_scaled = mfccs_scaled.reshape(1, -1)  # Reshape to match the model input
    return mfccs_scaled

# Predict genre from audio file
def predict_genre(model, file_path):
    mfccs_scaled = preprocess_audio(file_path)
    prediction = model.predict(mfccs_scaled)
    predicted_genre = GENRE_LABELS[np.argmax(prediction)]
    return predicted_genre

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sheet2music')
def s2m():
    return render_template('sheet2music.html')

@app.route('/music2sheet')
def m2s():
    return render_template('music2sheet.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/tools')
def tool():
    return render_template('tools.html')

# Route to handle audio classification
@app.route('/classify_audio', methods=['GET', 'POST'])
def classify_audio():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return "No file part"
        
        file = request.files['audio_file']
        if file.filename == '':
            return "No selected file"
        
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            try:
                predicted_genre = predict_genre(model, file_path)
                return render_template('classify_audio.html', prediction=predicted_genre)
            except Exception as e:
                print(f"Error: {e}")
                return "Error in processing the file. Prediction failed."
    
    return render_template('classify_audio.html')

# Add the audio file upload and analysis route
@app.route('/audio_visualization')
def audio_visualization():
    return render_template('audio_visualization.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return "No file part"
    file = request.files['audio']
    if file.filename == '':
        return "No selected file"
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Analyze the audio file
    waveform_image_path = generate_waveform(file_path)
    instruments_info = analyze_audio(file_path)

    return render_template('result.html', waveform=waveform_image_path, instruments=instruments_info)

def generate_waveform(file_path):
    y, sr = librosa.load(file_path)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    waveform_image_path = 'static/waveform.png'
    plt.savefig(waveform_image_path)
    plt.close()
    return waveform_image_path

def analyze_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Get pitches and magnitudes
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Get average pitch
    pitch = np.mean(pitches[pitches > 0])

    # Extract spectral features for potential instrument identification
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Use spectral features to guess instruments (simplified detection logic)
    if spectral_centroid > 3000 and spectral_bandwidth > 2000:
        instruments = 'Electric Guitar'
    elif 1000 < spectral_centroid < 3000:
        instruments = 'Piano'
    elif spectral_centroid < 1000:
        instruments = 'Bass Guitar or Cello'
    else:
        instruments = 'Unknown'

    # Simulate other audio properties
    audio_info = {
        'pitch': pitch,
        'instruments': instruments,
        'duration': librosa.get_duration(y=y, sr=sr),
        'sample_rate': sr
    }

    return audio_info

if __name__ == '__main__':
    app.run(debug=True)
