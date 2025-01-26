import os
import glob
import torch
import textgrid
from pydub import AudioSegment

# UD 기반 17개 품사 + (대명사 1/2인칭, 3인칭) + 침묵(SIL) = 총 19개
POS_LIST = [
    "ADJ",       # 형용사
    "ADP",       # 전치사/후치사
    "ADV",       # 부사
    "AUX",       # 조동사
    "CCONJ",     # 등위접속사
    "DET",       # 한정사
    "INTJ",      # 감탄사
    "NOUN",      # 명사
    "NUM",       # 수사
    "PART",      # 불변화사 등
    "PRON_1_2",  # 1,2인칭 대명사
    "PRON_3",    # 3인칭 대명사
    "PROPN",     # 고유명사
    "PUNCT",     # 구두점
    "SCONJ",     # 종속접속사
    "SYM",       # 기호
    "VERB",      # 동사
    "X",         # 기타/판단 불가
    "SIL"        # 침묵
]

POS2IDX = {pos: i for i, pos in enumerate(POS_LIST)}

# 1/2인칭 대명사 후보
FIRST_SECOND_PRONOUNS = {
    "i", "me", "myself", "we", "us", "our", "ourselves",
    "you", "your", "yourself", "yourselves",
    "we're", "you're"
}

# 3인칭 대명사 후보
THIRD_PRONOUNS = {
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "itself",
    "they", "them", "theirs", "themselves",
    "he's", "she's", "it's", "they're"
}

def normalize_pronoun(token: str) -> str:
    """
    대명사 축약형(he's, she's, it's 등)을 정규화.
    예: "he's" -> "he", "she's" -> "she" 등
    """
    t = token.lower().strip()
    expansions = {
        "he's": "he",
        "she's": "she",
        "it's": "it",
        "they're": "they",
        "you're": "you",
        "we're": "we"
    }
    if t in expansions:
        return expansions[t]
    return t

def map_pos(word_text: str, ner_tag: str) -> str:
    """
    word, ner 정보를 바탕으로 19개 품사 중 하나를 결정.
    침묵이면 SIL, 대명사면 1/2 vs 3인칭 분리.
    """
    # 빈 문자열 -> 침묵
    if not word_text.strip():
        return "SIL"
    
    word_lower = normalize_pronoun(word_text.lower().strip())
    pos_tag = ner_tag.upper()
    
    # 대명사 감별
    if pos_tag == "PRON":
        if word_lower in FIRST_SECOND_PRONOUNS:
            return "PRON_1_2"
        elif word_lower in THIRD_PRONOUNS:
            return "PRON_3"
        else:
            # 기타 대명사는 3인칭으로 처리(정책에 따라 조정)
            return "PRON_3"
    
    # POS_LIST 중 매칭
    for pos_candidate in POS_LIST:
        if pos_candidate.upper() == pos_tag:
            return pos_candidate
    
    # 없으면 X
    return "X"

def get_intervals_including_silence(tier: textgrid.IntervalTier, global_min: float, global_max: float):
    """
    주어진 tier에 대해 interval 사이 공백을 침묵으로 추가하여
    (start, end, mark) 형태의 리스트를 반환한다.
    """
    intervals = sorted(tier.intervals, key=lambda x: x.minTime)
    merged = []

    if not intervals:
        # 해당 tier에 interval이 하나도 없으면 전체 구간을 침묵 처리
        merged.append((global_min, global_max, ""))
        return merged

    # 맨 앞 침묵
    if intervals[0].minTime > global_min:
        merged.append((global_min, intervals[0].minTime, ""))  # 침묵

    # 중간 병합
    for i in range(len(intervals)):
        current = intervals[i]
        merged.append((current.minTime, current.maxTime, current.mark))
        
        # 다음 interval과 gap
        if i < len(intervals) - 1:
            nxt = intervals[i+1]
            if current.maxTime < nxt.minTime:
                # 침묵
                merged.append((current.maxTime, nxt.minTime, ""))

    # 맨 뒤 침묵
    if intervals[-1].maxTime < global_max:
        merged.append((intervals[-1].maxTime, global_max, ""))

    return merged

def process_textgrid_filelevel(textgrid_path: str, wav_path: str) -> torch.Tensor:
    """
    (파일 단위) : 하나의 TextGrid에 대해 품사별 카운트를 집계해
    38차원 텐서( 19차원: (카운트/단어수) + 19차원: (카운트/초) )를 반환.
    """
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    audio = AudioSegment.from_wav(wav_path)
    duration_sec = len(audio) / 1000.0
    if duration_sec == 0.0:
        duration_sec = 1e-6

    global_min = tg.minTime
    global_max = tg.maxTime

    word_tier = tg[0]
    ner_tier  = tg[1]

    w_intervals = get_intervals_including_silence(word_tier, global_min, global_max)
    n_intervals = get_intervals_including_silence(ner_tier,  global_min, global_max)

    count_pos = [0]*len(POS_LIST)
    total_words = 0
    
    for (w_start, w_end, w_text), (_, _, ner_tag) in zip(w_intervals, n_intervals):
        final_pos = map_pos(w_text, ner_tag)
        count_pos[POS2IDX[final_pos]] += 1
        
        if final_pos != "SIL":
            total_words += 1

    if total_words == 0:
        total_words = 1e-6

    vec_per_word = [c / total_words for c in count_pos]
    vec_per_sec  = [c / duration_sec  for c in count_pos]

    final_vec = vec_per_word + vec_per_sec  # 38차원
    return torch.tensor(final_vec, dtype=torch.float)

def process_textgrid_wordlevel(textgrid_path: str, wav_path: str) -> torch.Tensor:
    """
    (단어/interval 단위) : 하나의 TextGrid에 대해
    침묵 구간 포함 N개의 interval 각각을 19차원 원-핫 인코딩으로 만들어
    최종 (N x 19) 텐서를 반환.
    """
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    global_min = tg.minTime
    global_max = tg.maxTime
    
    word_tier = tg[0]
    ner_tier  = tg[1]

    w_intervals = get_intervals_including_silence(word_tier, global_min, global_max)
    n_intervals = get_intervals_including_silence(ner_tier,  global_min, global_max)

    # 각 interval마다 19차원 원-핫
    one_hot_list = []

    for (w_start, w_end, w_text), (_, _, ner_tag) in zip(w_intervals, n_intervals):
        final_pos = map_pos(w_text, ner_tag)
        
        # 19차원 중 해당 품사만 1, 나머지는 0
        vec = [0]*len(POS_LIST)
        pos_idx = POS2IDX[final_pos]
        vec[pos_idx] = 1
        
        one_hot_list.append(vec)

    return torch.tensor(one_hot_list, dtype=torch.float)  # (N x 19)

def main():
    all_data_38 = {}        # 파일 단위 38차원
    all_data_wordlevel = {} # (N x 19) 단어 단위
    
    # Audios/*.wav 전부 순회
    for wav_file in glob.glob("Audios/*.wav"):
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        tgrid_path = os.path.join("result_nertg", base_name + ".TextGrid")
        
        if os.path.exists(tgrid_path):
            # TextGrid와 WAV 파일이 모두 존재하는 경우
            
            # (1) 파일 단위 38차원
            feature_38 = process_textgrid_filelevel(tgrid_path, wav_file)
            all_data_38[base_name] = feature_38
            
            # (2) 단어 단위 Nx19
            word_tensor = process_textgrid_wordlevel(tgrid_path, wav_file)
            all_data_wordlevel[base_name] = word_tensor
        
        else:
            # TextGrid 없으면 모두 0으로 처리
            # 38차원 0벡터
            all_data_38[base_name] = torch.zeros(38, dtype=torch.float)
            # 단어 단위: 구간 자체가 없으니 (1 x 19) 또는 (0 x 19) 중 선택 가능
            # 여기서는 (0 x 19) 텐서로 처리
            all_data_wordlevel[base_name] = torch.zeros(0, 19, dtype=torch.float)

    # 결과 저장
    torch.save(all_data_38, "word_pos_filelevel.pt")
    torch.save(all_data_wordlevel, "word_pos_wordlevel.pt")

    print("Saved word_pos_filelevel.pt and word_pos_wordlevel.pt")

if __name__ == "__main__":
    main()
