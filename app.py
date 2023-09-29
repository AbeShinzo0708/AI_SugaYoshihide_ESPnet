import time

import streamlit as st
import numpy as np
import torch
from espnet2.bin.tts_inference import Text2Speech
from scipy.io.wavfile import write
from PIL import Image


fs, lang = 44100, "Japanese"
model= "./100epoch.pth"
x = "これはテストメッセージです"

text2speech = Text2Speech.from_pretrained(
    model_file=model,
    device="cpu",
    speed_control_alpha=1.0,
    noise_scale=0.333,
    noise_scale_dur=0.333,
)
pause = np.zeros(30000, dtype=np.float32)

st.title("おしゃべりAI菅義偉メーカー")
image = Image.open('suga.jpg')
st.image(image)
text = st.text_area(label='ここにテキストを入力 (Input Text)↓', height=100, max_chars=2048)


if st.button("生成（Generate）"):
    with torch.no_grad():
        wav = text2speech(text)["wav"]

    wav_list = []
    wav_list.append(np.concatenate([wav.view(-1).cpu().numpy(), pause]))
    final_wav = np.concatenate(wav_list)
    st.audio(final_wav, sample_rate=fs)
