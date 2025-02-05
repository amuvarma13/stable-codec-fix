import torch
import torchaudio
from stable_codec import StableCodec
import soundfile as sf

model = StableCodec(
    pretrained_model = 'stabilityai/stable-codec-speech-16k', 
    device = torch.device("cuda")
)

audiopath = "fixed_low-f.wav"
model.set_posthoc_bottleneck("1x46656_400bps")
latents, tokens = model.encode(audiopath, posthoc_bottleneck = True)
decoded_audio = model.decode(tokens, posthoc_bottleneck = True)

#save the audio
sf.write("decoded_audio.wav", decoded_audio.cpu().squeeze(), 16000)