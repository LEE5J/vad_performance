import tgt
from glob import glob
from sgvad import SGVAD


if __name__ == "__main__":
    sgvad = SGVAD.init_from_ckpt()
    va_set = {}
    empty_tg = []
    frame_sec = 0.1
    total_score = 0
    total_frame = 0
    total_miss = 0
    for filepath in glob("ForceAligned/*"): 
        voice_active = []
        try: 
            tg = tgt.read_textgrid(filepath)
        except IndexError as e:
            empty_tg.append(filepath)
            continue
        tg = tg.tiers[0] 
        for item in tg:
            voice_active.append([item.start_time,item.end_time]) 
        va_set['filepath'] = voice_active
        # vad 확인하고 
        base_file = filepath.split('/')[-1].replace(".TextGrid",'')
        pred = sgvad.predict_frame_context(f"Audios/{base_file}.wav",frame_sec=frame_sec)
        score = 0
        miss = 0
        num_frames = len(pred)
        labels = [0] * num_frames
        for period in voice_active:
            start = int(float(period[0])*(1/frame_sec))
            end = int(float(period[1])*(1/frame_sec))
            for i in range(start,end):
                labels[i] = 1
        for i in range(num_frames):
            if pred[i]==labels[i]:
                score += 1
            else:
                miss += pred[i]-labels[i]
        total_score += score
        total_frame += num_frames
        total_miss += miss
        # vad 일치도 계산
        print(total_score/total_frame*100)
        print("is sensivie?",miss/total_frame*100)

