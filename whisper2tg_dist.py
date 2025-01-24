import os
import glob
import whisper
import textgrid
from pydub import AudioSegment
from tqdm import tqdm
import torch

# TextGrid 파일 생성 함수
def create_textgrid(file_path, result, output_dir):
    audio = AudioSegment.from_wav(file_path)
    duration_seconds = len(audio) / 1000.0

    tg = textgrid.TextGrid()
    word_tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=duration_seconds)

    pad = 0.0
    for segment in result["segments"]:
        for word_info in segment["words"]:
            text = word_info["word"].strip()
            if text:  # 공백 제외
                start_time = word_info["start"] + pad
                pad = 0.0
                end_time = word_info["end"]
                while start_time >= end_time:
                    end_time += 0.01
                    pad += 0.01
                word_tier.addInterval(
                    textgrid.Interval(start_time, end_time, text)
                )

    tg.append(word_tier)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.TextGrid")
    tg.write(output_path)
    print(f"Saved TextGrid: {output_path}")

# 작업 처리 함수 (GPU에서 실행)
def transcribe_and_save(model, file_path, output_dir):
    result = model.transcribe(
        file_path,
        word_timestamps=True,
        beam_size=5,
        patience=1,
        temperature=0.0,
        language="en"
    )
    create_textgrid(file_path, result, output_dir)

# 메인 함수: 특정 GPU 및 데이터 분할 처리
def main():
    # 사용자로부터 GPU ID 입력받기
    gpu_id = int(input("Enter GPU ID (0-2): "))
    if gpu_id < 0 or gpu_id > 2:
        raise ValueError("Invalid GPU ID! Please enter a number between 0 and 2.")

    # CUDA_VISIBLE_DEVICES 환경변수 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using GPU: {gpu_id} (mapped to {device})")

    # 데이터셋 경로 및 출력 경로 설정
    DIR_PATH = "Audios"
    TARG_PATH = "result_tg"
    os.makedirs(TARG_PATH, exist_ok=True)

    # 오디오 파일 정렬 및 분할
    audio_files = sorted(glob.glob(os.path.join(DIR_PATH, "*.wav")))
    num_files = len(audio_files)
    
    # 데이터를 4등분하여 현재 GPU에 해당하는 부분만 가져오기
    files_per_gpu = num_files // 3 + (num_files % 3 > 0)  # 각 GPU에 할당될 파일 수 계산
    start_idx = gpu_id * files_per_gpu                   # 현재 GPU의 시작 인덱스
    end_idx = min(start_idx + files_per_gpu, num_files)  # 현재 GPU의 끝 인덱스
    assigned_files = audio_files[start_idx:end_idx]

    print(f"Assigned {len(assigned_files)} files to GPU {gpu_id}")

    # Whisper 모델 로드 (한 번만)
    print(f"Loading Whisper model on {device}")
    model = whisper.load_model("turbo", device=device)

    # 파일 처리 루프
    for file_path in tqdm(assigned_files):
        try:
            transcribe_and_save(model, file_path, TARG_PATH)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()
