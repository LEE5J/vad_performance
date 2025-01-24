import glob
import torch
from omegaconf import OmegaConf, DictConfig
from nemo.collections.asr.modules import AudioToMFCCPreprocessor, ConvASREncoder
import librosa


class SGVAD:
    def __init__(self, preprocessor: AudioToMFCCPreprocessor,model: ConvASREncoder,cfg: DictConfig):
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.preprocessor.featurizer.dither = 0.0
        self.preprocessor.featurizer.pad_to = 0
        self.model = model
        self.model.eval()
        self.model.freeze()
        self.preprocessor.to(self.cfg.device)
        self.model.to(self.cfg.device)

    def predict(self, wave):
        if isinstance(wave, str):
            wave = self.load_audio(wave)
            wave = torch.tensor(wave)
        if not isinstance(wave, torch.Tensor):
            wave = torch.tensor(wave)
        wave = wave.reshape(1, -1).to(self.cfg.device)
        wave_len = torch.tensor([wave.size(-1)]).reshape(1).to(self.cfg.device)
        processed_signal, processed_signal_len = self.preprocessor(input_signal=wave, length=wave_len)
        with torch.no_grad():
            mu, _ = self.model(audio_signal=processed_signal, length=processed_signal_len)
            binary_gates = torch.clamp(mu + 0.5, 0.0, 1.0)
            score = binary_gates.sum(dim=1).mean().item()
        return score >= self.cfg.threshold

    def load_audio(self, fpath):
        return librosa.load(fpath, sr=self.cfg.sample_rate)[0]

    @classmethod
    def init_from_ckpt(cls):
        cfg = OmegaConf.load("./cfg.yaml")
        ckpt = torch.load(cfg.ckpt, map_location='cuda')
        preprocessor = AudioToMFCCPreprocessor(**cfg.preprocessor)
        preprocessor.load_state_dict(ckpt['preprocessor'], strict=True)
        vad = ConvASREncoder(**cfg.vad)
        vad.load_state_dict(ckpt['vad'], strict=True)
        return cls(preprocessor, vad, cfg)

    def save_ckpt(self):
        ckpt_dict = {"preprocessor": self.preprocessor.state_dict(), "vad": self.model.state_dict()}
        torch.save(ckpt_dict, './sgvad.pth')

    def predict_frame_context(self, audio_path,frame_sec=0.1,width_rate=1):
        wave_data = self.load_audio(audio_path)
        sr = self.cfg.sample_rate
        wave_tensor = torch.tensor(wave_data)
        wave_len = len(wave_tensor)

        # 20ms
        chunk_len = int(sr * frame_sec)  # 예: sr=16000 → 320 샘플
        # 전체에서 20ms 단위로 나눴을 때, 남은 구간이 chunk_len(frame_sec) 미만이면 버린다
        num_chunks = wave_len // chunk_len
        gap = chunk_len * width_rate

        results = []
        for i in range(num_chunks):
            # 중앙 구간 (실제 라벨 뽑을 frame_sec)
            center_start = i * chunk_len
            center_end = center_start + gap

            if (i == 0):
                # 첫 구간: 앞쪽 frame_sec가 없으므로 2frame_secms (frame_sec + 뒤 frame_sec)
                start = center_start
                center_end = center_start + gap
                if end > wave_len:
                    end = wave_len
            else:
                # 나머지 구간: 앞뒤 frame_sec 씩 총 3 frame_sec
                start = center_start - gap
                end = center_end + gap
                if end > wave_len:
                    end = wave_len
            # 실제 예측용 구간
            chunk = wave_tensor[start:end]
            if len(chunk) == 0:
                continue
            label = self.predict(chunk)
            # 시간 정보(초 단위)
            results.append(label)
        return results