import torch
from tqdm import tqdm
import whisper
import glob,os
import textgrid
from pydub import AudioSegment



model = whisper.load_model("turbo")
DIR_PATH = "Audios"
audio_files = sorted(glob.glob(os.path.join(DIR_PATH, "*.wav")))
# for file_path in audio_files: # for debug
#     result = model.transcribe(
#         file_path,
#         word_timestamps=True,   # 단어 단위 타임스탬프 활성화
#         beam_size=5,
#         patience=1,
#         temperature=0.0,
#         language="en"
#     )
#     for segment in result["segments"]:
#         print(f"문장: {segment['text']}")
#         print(f"문장 시작: {segment['start']}, 문장 끝: {segment['end']}")
#         for word_info in segment["words"]:
#             print(
#                 f"  단어: {word_info['word']}, "
#                 f" 시작: {word_info['start']}, 끝: {word_info['end']}"
#             )
#         print()



for file_path in tqdm(audio_files):
    result = model.transcribe(
        file_path,
        word_timestamps=True,
        beam_size=5,
        patience=1,
        temperature=0.0,
        language="en"
    )

    # 오디오 정보 불러오기 (pydub 활용)
    audio = AudioSegment.from_wav(file_path)
    duration_seconds = len(audio) / 1000.0

    # TextGrid 생성
    tg = textgrid.TextGrid()
    word_tier = textgrid.IntervalTier(
        name="words",
        minTime=0.0,
        maxTime=duration_seconds
    )

    # Whisper 결과에서 단어 정보만 추출해 Tier 추가
    pad = 0.0
    for segment in result["segments"]:
        for word_info in segment["words"]:
            text = word_info["word"].strip()
            if text :  # 공백 제외
                start_time = word_info["start"] + pad
                pad =0.0
                end_time = word_info["end"]
                while start_time >= end_time:
                    end_time += 0.05
                    pad += 0.05
                word_tier.addInterval(
                    textgrid.Interval(start_time, end_time, text)
                )

    tg.append(word_tier)

    # TextGrid 파일로 저장
    TARG_PATH = 'result_tg'
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(TARG_PATH, f"{base_name}.TextGrid")
    tg.write(output_path)