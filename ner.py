import glob
import os
import sys  # 새로 추가한 부분
import textgrid
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import whisper
from pydub import AudioSegment


def make_pos_pairs(total_words, pred_pos1):
    """
    Whisper 토큰화 결과인 total_words를 기준으로,
    pred_pos에서 나온 품사가 만약 서브토큰으로 쪼개져 있으면
    첫 서브토큰의 품사를 사용하고 나머지는 무시하는 식으로
    (word, pos) 페어를 만든다.
    """
    pos_index = 0
    pairs = []
    pred_pos2 = [] # whisper 에서는 2개지만 품사태깅에서 하나로 묶어 버린것을 분리
    for i in range(len(pred_pos1)):
        pos = pred_pos1[i]
        if " " in pos['word']:
            pos_parts = pos['word'].split(' ')
            for parts in pos_parts:
                pred_pos2.append({'entity_group':pos['entity_group'],'word':parts})
        else:
            pred_pos2.append({'entity_group':pos['entity_group'],'word':pos['word']})
    pred_pos3 = [] # <unk> 제거
    for i in range(len(pred_pos2)):
        pos = pred_pos2[i]
        if '<unk>' == pos['word'] :
            continue
        if '<unk>' in pos['word'] :
            pred_pos3.append({'entity_group':pos['entity_group'],'word':pos['word'].replace("<unk>","")})
            continue
        if "@@" in pos['word'] and i+1 < len(pred_pos2) and pred_pos2[i+1]['word'] == '<unk>':
            pred_pos3.append({'entity_group':pos['entity_group'],'word':pos['word'].replace("@@","")})
            continue
        pred_pos3.append(pos)
    pred_pos4 = [] # @@ 이어 붙이기
    i = 0
    while i < len(pred_pos3):
        pos = pred_pos3[i]
        first = {'entity_group':pos['entity_group'],'word':pos['word'].replace("@@","")}
        while "@@" in pos['word']:
            i += 1
            pos = pred_pos3[i]
            first = {'entity_group':first['entity_group'],'word':first['word']+pos['word'].replace("@@","")}
        pred_pos4.append(first)
        i += 1
    pred_pos = pred_pos4

    # 순회하면서 맞춰보기
    for w in total_words:
        first_pos = pred_pos[pos_index]['entity_group']
        # 여기서 일치가 아니라 in 을 쓰는 이유는 <unk> 때문에 손실있음
        if pred_pos[pos_index]['word'] not in w['word']: # mismatch
            raise ValueError()
        pairs.append((w, first_pos))
        pos_index += 1
    if len(pairs)!=len(total_words):
        raise IndexError
    return pairs


def transcribe(model, file_path):
    result = model.transcribe(
        file_path,
        word_timestamps=True,
        beam_size=5,
        patience=1,
        temperature=0.0,
        language="en"
    )
    return result


if __name__ == "__main__":
    # sys.argv에서 GPU 아이디(0, 1, 2 중 하나)를 입력받는다.
    # 예: python name.py 0
    gpu_id = int(sys.argv[1])  # 0, 1, 2
    #gpu_id = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Using GPU {gpu_id}")
    # 모델 불러오기
    model_name = "TweebankNLP/bertweet-tb2_ewt-pos-tagging"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_pos = AutoModelForTokenClassification.from_pretrained(model_name)
    pos_pipeline = pipeline(
        task="token-classification",
        model=model_pos,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    DIR_PATH = "Audios"
    audio_files = sorted(glob.glob(os.path.join(DIR_PATH, "*.wav")))
    num_files = len(audio_files)

    # 전체 파일을 3등분해서, GPU 아이디에 맞는 구간만 처리하도록 함
    chunk_size = num_files // 3
    start_idx = gpu_id * chunk_size 
    # 마지막 GPU일 경우에는 나머지도 포함하기 위해 num_files까지 슬라이싱
    end_idx = num_files if gpu_id == 2 else (gpu_id + 1) * chunk_size
    sub_audio_files = audio_files[start_idx:end_idx]

    # Whisper 모델 불러오기
    device = f'cuda'
    model_whisper = whisper.load_model("turbo", device=device)

    error_list = []
    for file_path in tqdm(sub_audio_files):
        try:
            total_words = []
            trsc = transcribe(model_whisper, file_path)
            for sentence in trsc['segments']:
                words_in_sentence = sentence["words"]
                for word in words_in_sentence:
                    total_words.append(word)
            whole_sentence = ""
            for word in total_words:
                whole_sentence += word['word'] + " "

            pred_pos = pos_pipeline(whole_sentence)
            try:
                pairs = make_pos_pairs(total_words, pred_pos)
            except ValueError:
                error_list.append(file_path)
                print("error path", file_path)
                continue
            except IndexError:
                error_list.append(file_path)
                print("length miss match", file_path)
                continue
            
            min_gap = 0.03
            # 오디오 정보 불러오기 (pydub 활용)
            audio = AudioSegment.from_wav(file_path)
            duration_seconds = len(audio) / 1000.0
            # 오디오를 연장해야할 수도 있으니 고려하여 연장
            pad = 0.0
            for segment in trsc["segments"]:
                for word_info in segment["words"]:
                    start_time = word_info["start"] + pad
                    pad = 0.0
                    end_time = word_info["end"]
                    # start와 end가 같을 경우 살짝 늘림
                    if start_time >= end_time:
                        end_time += min_gap
                        pad += min_gap
                        if start_time == end_time:
                            end_time += 0.01
                            pad += 0.01
                        if end_time > duration_seconds:
                            duration_seconds = end_time
                        
            
            # TextGrid 생성
            tg = textgrid.TextGrid()
            word_tier = textgrid.IntervalTier(
                name="words",
                minTime=0.0,
                maxTime=duration_seconds
            )
            # Whisper 결과에서 단어 정보만 추출해 Tier 추가
            pad = 0.0
            timestamps = []
            for segment in trsc["segments"]:
                for word_info in segment["words"]:
                    text = word_info["word"].strip()
                    if text:  # 공백 제외
                        start_time = word_info["start"] + pad
                        pad = 0.0
                        end_time = word_info["end"]
                        # start와 end가 같을 경우 살짝 늘림
                        if start_time >= end_time:
                            end_time += min_gap
                            pad += min_gap
                            if start_time == end_time:
                                end_time += 0.01
                                pad += 0.01
                        timestamps.append([start_time, end_time])
                        word_tier.addInterval(
                            textgrid.Interval(start_time, end_time, text)
                        )
            tg.append(word_tier)

            ner_tier = textgrid.IntervalTier(
                name="ner",
                minTime=0.0,
                maxTime=duration_seconds
            )
            for i in range(len(timestamps)):
                start_time, end_time = timestamps[i]
            
                ner_tier.addInterval(
                    textgrid.Interval(start_time, end_time, pairs[i][1])
                ) 
                
                        
            tg.append(ner_tier)

            TARG_PATH = 'result_nertg'
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(TARG_PATH, f"{base_name}.TextGrid")
            tg.write(output_path)
        except:
            print("tg error",file_path)

    if error_list:
        print("\nError files:")
        for ef in error_list:
            print(ef)