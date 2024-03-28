import torchaudio
from pyannote.audio import Pipeline
import openai_whisper

# Initialize the diarization pipeline
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Analyze the audio file to identify speakers and their segments
audio_file_path = "path/to/your/audio/file.wav"
diarization = diarization_pipeline(audio_file_path)

# Initialize Whisper
model = openai_whisper.load_model("base")

def transcribe_segment(audio_segment):
    # Convert the audio segment to the expected format for Whisper
    audio_segment = audio_segment.mean(dim=0, keepdim=True)  # Convert to mono
    result = model.transcribe(audio_segment.numpy())
    return result["text"]

# Load the entire audio file
audio, sample_rate = torchaudio.load(audio_file_path)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_sample = int(turn.start * sample_rate)
    end_sample = int(turn.end * sample_rate)
    audio_segment = audio[:, start_sample:end_sample]

    # Transcribe the audio segment
    transcription = transcribe_segment(audio_segment)

    print(f"Speaker {speaker}: {transcription}")

