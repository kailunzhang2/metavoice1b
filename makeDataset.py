# whisperx "/home/eshoyuan/Documents/voice_clone/data/Trump WEF 2018.wav" --model small --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4
from pydub import AudioSegment
import pandas as pd
import re
import os

def parse_srt(srt_file_path):
    # Parse srt file
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        srt_content = file.read()
    pattern = re.compile(r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', re.DOTALL)
    return [(m.group(1), m.group(2), m.group(3).replace('\n', ' ')) for m in pattern.finditer(srt_content)]

def srt_time_to_millis(srt_time):
    # Convert srt time to milliseconds
    hours, minutes, seconds = map(float, srt_time[:-4].split(':'))
    milliseconds = int(srt_time[-3:])
    return int((hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds)

def split_wav_srt(wav_file_path, srt_file_path, output_dir):
    # Split wav file based on srt file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio = AudioSegment.from_wav(wav_file_path)
    segments = parse_srt(srt_file_path)
    mapping = []
    start_millis = None
    cached_caption = "" # Cache caption for segments less than 30 seconds
    for index, (start, end, caption) in enumerate(segments):
        if start_millis == None:
            start_millis = srt_time_to_millis(start)
        end_millis = srt_time_to_millis(end)
        cached_caption += caption + " "
        if end_millis - start_millis < 30000:
            continue
        segment = audio[start_millis:end_millis]
        audio_file_name = f"segment_{index}.wav"
        start_millis = None
        segment.export(os.path.join(output_dir, audio_file_name), format="wav")
        text_file_name = f"segment_{index}.txt"
        with open(os.path.join(output_dir, text_file_name), 'w', encoding='utf-8') as text_file:
            text_file.write(cached_caption)
            cached_caption = ""
        mapping.append({"audio_files": output_dir+audio_file_name, "captions": output_dir+text_file_name})
    return mapping

def create_csv(mapping, output_file_path):
    df = pd.DataFrame(mapping)
    df.to_csv(output_file_path, index=False, sep='|')

# Example usage
wav_file_path = 'path/to/preprocessed/audio/files'
srt_file_path = 'path/to/srt/files'
output_dir = 'path/to/output/directory'
csv_file_path = 'path/to/csv/file'

mapping = split_wav_srt(wav_file_path, srt_file_path, output_dir)
create_csv(mapping, csv_file_path)
