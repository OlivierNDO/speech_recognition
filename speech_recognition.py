
# Imports
from flask import Flask, request, render_template, redirect, url_for
import torch
import os
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, MarianMTModel, MarianTokenizer
from langdetect import detect
import warnings


# Function to load MP3 and convert it into the required format
def load_mp3(mp3_file: str, n_seconds: int = None):
    """
    Load MP3 file and optionally truncate to the first n seconds.
    
    Parameters
    ----------
    mp3_file : str
        Path to the MP3 file.
    n_seconds : int, optional
        Number of seconds to load from the beginning of the file. 
        If None, load the entire file.
    
    Returns
    -------
    waveform : np.ndarray
        Audio waveform.
    sample_rate : int
        Sample rate of the audio.
    """
    # Load the MP3 file
    if n_seconds is not None:
        # Calculate the number of samples to load (sample_rate * n_seconds)
        waveform, sample_rate = librosa.load(mp3_file, sr=16000, duration=n_seconds)
    else:
        # Load the entire file
        waveform, sample_rate = librosa.load(mp3_file, sr=16000)
    
    return waveform, sample_rate


# Language dictionary for detection
language_dict = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan", 
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", "en": "English", 
    "es": "Spanish", "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French", 
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", 
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kn": "Kannada", "ko": "Korean", 
    "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", 
    "ne": "Nepali", "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish", 
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", 
    "so": "Somali", "sq": "Albanian", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", 
    "te": "Telugu", "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", 
    "ur": "Urdu", "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)"
}


class NicksLanguageDetector:
    def __init__(self, model_id='openai/whisper-large-v2'):
        self.model_id = model_id
        self._load_model()
        self._load_processor()
        self._load_pipelines()


    def _load_model(self):
        """Load Whisper model"""
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)


    def _load_processor(self):
        """Load data processor"""
        self.processor = AutoProcessor.from_pretrained(self.model_id)


    def _load_pipelines(self):
        """Load the transcription pipeline"""
        self.transcription_pipeline = pipeline(
            'automatic-speech-recognition',
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )


    def transcribe(self, mp3_file_path: str):
        """Transcribe audio in the original language"""
        waveform, sample_rate = load_mp3(mp3_file_path, 8)
        transcription = self.transcription_pipeline(waveform)['text']
        return transcription


    def detect_and_transcribe(self, mp3_file_path: str):
        """Transcribe and detect language of the audio"""
        try:
            print(f"Starting transcription for: {mp3_file_path}")
            transcription = self.transcribe(mp3_file_path)
            print(f"Transcription completed: {transcription}")

            detected_language_code = detect(transcription)
            detected_language = language_dict.get(detected_language_code, "Unknown")
            print(f"Detected Language: {detected_language}")

            return detected_language, transcription
        except Exception as e:
            print(f"Error in detect_and_transcribe: {e}")
            return None




# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Create an instance of the language detector and test it
detector = NicksLanguageDetector()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file to the upload folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Detect language and transcribe the audio
            detected_language, transcription = detector.detect_and_transcribe(filepath)

            # Render the result page with the detected language and transcription
            return render_template('result.html', 
                                   language=detected_language, 
                                   transcription=transcription)

    return render_template('upload.html')

# Define a simple upload form in HTML
@app.route('/upload.html')
def upload_form():
    return """
    <html>
        <body>
            <h1>Upload an MP3 File for Transcription</h1>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

# Define the result page in HTML
@app.route('/result.html')
def result():
    return """
    <html>
        <body>
            <h1>Transcription Result</h1>
            <p><strong>Detected Language:</strong> {{ language }}</p>
            <p><strong>Transcription:</strong></p>
            <p>{{ transcription }}</p>
            <a href="/">Upload another file</a>
        </body>
    </html>
    """


if __name__ == '__main__':
    app.run(debug=True)
