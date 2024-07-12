import importlib
from pathlib import Path
import sys
import os
import time

sys.path.append("C:\\Users\\Alafate\\Documents\\TTS\\")
print(sys.path)
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# from TTS.api import TTS
import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt

# tts_path = "C:\\Users\\User\\AppData\\Local\\tts\\tts_models--en--vctk--vits\\model_file.pth"
# tts_config_path = "C:\\Users\\User\\AppData\\Local\\tts\\tts_models--en--vctk--vits\\config.json"
# speakers_file_path = "C:\\Users\\User\\AppData\\Local\\tts\\tts_models--en--vctk--vits\\speakers.json"
# tts_path = "path/to/xtts"

# model = "tts_models/en/ljspeech/vits--neon"
# model = "tts_models/en/vctk/vits"
# # model = "tts_models/multilingual/multi-dataset/xtts_v2"

# # module = importlib.import_module("TTS.tts.configs.vits_config")

# tts = TTS(model).to("cuda")
# synthesizer = tts.synthesizer

# md = Path(__file__).parent.parent.parent / "TTS/.models.json"
# print("md = ", md)
# manager = ModelManager(models_file=md, progress_bar=True, verbose=False)
# output_model_path, output_config_path, model_item = manager.download_model(model)
# synthesizer = Synthesizer(
#     tts_checkpoint=output_model_path,
#     tts_config_path=output_config_path,
#     # tts_speakers_file=speakers_file_path,
#     use_cuda=True,
# )

# synthesizer = Synthesizer(
#     tts_checkpoint=tts_path,
#     tts_config_path=tts_config_path,
#     tts_speakers_file=speakers_file_path,
#     use_cuda=True,
# )

# synthesizer = Synthesizer(
#     tts_checkpoint=model_path,
#     tts_config_path=config_path,
#     tts_speakers_file=None,
#     tts_languages_file=None,
#     vocoder_checkpoint=vocoder_path,
#     vocoder_config=vocoder_config_path,
#     encoder_checkpoint=None,
#     encoder_config=None,
#     use_cuda=gpu,
# )

# print(synthesizer)

# raw_text = "Hey ! How are you my dear ? it's sunny today."
# # raw_text = "hello world this is a demonstration sentence."
# wavs, outputs = tts.tts(text=raw_text, speaker="p225", return_extra_outputs=True)
# # wavs, outputs = synthesizer.tts(text=raw_text, speaker_name="p225", return_extra_outputs=True)
# print(len(wavs))
# print(len(outputs))
# print(outputs.keys())
# print(len(outputs["wav"]))
# print(outputs["alignments"].size())
# print(len(outputs["text_inputs"]))
# print(outputs["outputs"].keys())
# print(outputs["outputs"]["model_outputs"].size())
# print(outputs["outputs"]["alignments"].size())
# print(outputs["outputs"]["durations"].size())


# tokens = synthesizer.tts_model.tokenizer.text_to_ids(raw_text)

# pre_tokenized_txt = [synthesizer.tts_model.tokenizer.decode([y]) for y in tokens]
# # replace <blnk> with space
# pre_tokenized_text = [x if x != "<BLNK>" else "_" for x in pre_tokenized_txt]

# print(len(raw_text))
# print(len(tokens))
# print(len(pre_tokenized_txt))
# print(len(pre_tokenized_text))
# print(tokens)
# print(pre_tokenized_txt)
# print(pre_tokenized_text)

# # durations = np.array(outputs["outputs"]["durations"])

# # len of tokens is equal to the number of phonemes in the sentence which is also equal to the number of durations.
# nb_tokens = len(pre_tokenized_text)
# duration_in_seconds = len(wavs) / 22050
# duration_in_phonems = outputs["outputs"]["durations"].size()
# total_duration = int(outputs["outputs"]["durations"].sum().item())

# print("nb_tokens = ", nb_tokens)
# print("duration_in_seconds = ", duration_in_seconds)
# print("duration_in_phonems = ", duration_in_phonems)
# print("total_duration = ", total_duration)
# print("pre_tokenized_text = ", pre_tokenized_text)


# ### MARIUS BEGINS

# list_words_token = []
# list_words = []
# last_id = 0
# for i, token in enumerate(pre_tokenized_text):
#     if token == " " or i == len(pre_tokenized_text) - 1:
#         list_words_token.append(pre_tokenized_text[last_id : i + 1])
#         list_words.append("".join(list_words_token[-1]))
#         last_id = i + 1
# len_words = [len(w) for w in list_words_token]
# print("list_words = ", list_words)
# print("len words = ", len_words)
# print("total len words = ", sum(len_words))

# assert sum(len_words) == nb_tokens

# duration_words = []
# p_list = []
# old_len_w = 0
# for len_w in len_words:
#     duration_words.append(int(outputs["outputs"]["durations"][0][0][old_len_w : old_len_w + len_w].sum().item()))
#     p_list.append(int(len(outputs["wav"]) * duration_words[-1] / total_duration))
#     old_len_w = old_len_w + len_w

# print(outputs["outputs"]["durations"][0][0])
# print("duration_words = ", duration_words)
# print("p_list = ", p_list)
# print("total duration_words = ", sum(duration_words))
# print("total p_list = ", sum(p_list))

# # The difference between outputs["wavs"] and wavs is that
# # in wav there is an additional 10000 frames of silence at the end


# assert sum(duration_words) == total_duration
# assert sum(p_list) == len(outputs["wav"])


# import pyaudio

# p = pyaudio.PyAudio()
# stream = p.open(format=p.get_format_from_width(2), channels=1, rate=22050, input=False, output=True)

# old_p = 0
# for p in p_list:
#     waveforms = outputs["wav"][old_p : old_p + int(p)]
#     waveforms = np.array(waveforms)
#     waveforms = (waveforms * 32767).astype(np.int16).tobytes()
#     print(len(waveforms))
#     old_p = old_p + int(p)

#     stream.write(waveforms)

#     time.sleep(2)

# full_waveforms = outputs["wav"]
# full_waveforms = np.array(full_waveforms)
# full_waveforms = (full_waveforms * 32767).astype(np.int16).tobytes()
# print(len(full_waveforms))

# stream.write(full_waveforms)

# full_waveforms = wavs
# full_waveforms = np.array(full_waveforms)
# full_waveforms = (full_waveforms * 32767).astype(np.int16).tobytes()
# print(len(full_waveforms))

# stream.write(full_waveforms)

# waveforms = outputs["wav"][: int(p_2.item())]
# waveforms = np.array(waveforms)
# waveforms = (waveforms * 32767).astype(np.int16).tobytes()
# print(len(waveforms))

# stream.write(waveforms)


# TODO: check if you can distinguish words from the durations, so that we can get words timestamps from tts.


# # plot the spectrogram
# spec = synthesizer.tts_model.ap.melspectrogram(np.array(wavs))
# plt.figure(figsize=(20, 5))
# plt.imshow(spec, origin="lower", aspect="auto", interpolation="none")
# # create xticks with pre_tokenized_text
# plt.show()


##########
# the shortest way to get each word chunk

# imports
import sys
import os
import time

sys.path.append("C:\\Users\\Alafate\\Documents\\TTS\\")
print(sys.path)
from TTS.api import TTS
import pyaudio

# parameters
SPACE_TOKEN_ID = 16
NB_FRAME_PER_DURATION = 256
model = "tts_models/en/vctk/vits"
raw_text = "hello world this is a demonstration sentence."
# raw_text = "Hey ! How are you my dear ? it's sunny today."

# init
tts = TTS(model).to("cuda")
synthesizer = tts.synthesizer
tokens = synthesizer.tts_model.tokenizer.text_to_ids(raw_text)
wavs, final_outputs = tts.tts(text=raw_text, speaker="p225", return_extra_outputs=True)
# wavs, final_outputs = synthesizer.tts(text=raw_text, speaker_name="p225", return_extra_outputs=True)

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(2), channels=1, rate=22050, input=False, output=True)

for outputs in final_outputs:

    # intermediate parameters
    space_tokens_ids = [i + 1 for i, x in enumerate(tokens) if x == SPACE_TOKEN_ID or i == len(tokens) - 1]
    len_wav = len(outputs["wav"])
    durations = outputs["outputs"]["durations"].squeeze().tolist()
    total_duration = int(sum(durations))

    wav_words_chunk_len = []
    old_len_w = 0
    for s_id in space_tokens_ids:
        wav_words_chunk_len.append(int(sum(durations[old_len_w:s_id])) * NB_FRAME_PER_DURATION)
        # wav_words_chunk_len.append(int(sum(durations[old_len_w:s_id])) * len_wav / total_duration )
        old_len_w = s_id

    # Play audio to test if the words are correctly divided.
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2), channels=1, rate=22050, input=False, output=True)
    old_chunk_len = 0
    for chunk_len in wav_words_chunk_len:
        waveforms = outputs["wav"][old_chunk_len : old_chunk_len + int(chunk_len)]
        waveforms = np.array(waveforms)
        waveforms = (waveforms * 32767).astype(np.int16).tobytes()
        old_chunk_len = old_chunk_len + int(chunk_len)
        stream.write(waveforms)
        time.sleep(2)
