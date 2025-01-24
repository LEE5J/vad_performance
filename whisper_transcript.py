import glob
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import textgrid
from tqdm import tqdm


device = "cuda:0"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"



model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


# Save Directories
TRANSCRIPT_SAVE_DIR = "Transcripts_test"
CHUNK_SAVE_DIR = "ForceAligned_test"
for d in [TRANSCRIPT_SAVE_DIR, CHUNK_SAVE_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Data Path
DIR_PATH = "Audios"

audio_files = sorted(glob.glob(os.path.join(DIR_PATH, "*.wav")))[:100]

result = pipe(DIR_PATH, batch_size=8, return_timestamps='word', generate_kwargs={"language": "english"})


