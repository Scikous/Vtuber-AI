# Vtuber-AI
This project is meant to give a good starting point for others to create their own custom AI Vtubers (technically won't have to be a Vtuber). 

# Prompt Style
> Dataset style is in LLM training section (WIP)
```
<|im_start|>system: character behaviour <|im_end|> <|im_start|>context: information about current situation (previous dialogue or description of situation) <|im_end|> <|im_start|>user: Question or Instruction <|im_end|> <|im_start|>assistant: Model Answer
```
# Virtual Environments
The virtual environment simply helps to avoid package conflicts. Do note that this will take more space in the storage as each environment is its own.

:information_source: Note that this is for CMD

Create env (the last `venv` is the folder name/path where the venv will be created):
```
 python -m venv venv
```

Activate env:
```
venv\Scripts\activate.bat
```

Deactivate env: ```deactivate```
Then just delete the venv folder

# Voice Model
## Training
### Official Guide: 
follow the user guide from: https://github.com/RVC-Boss/GPT-SoVITS/tree/main

user guide direct link (can change at any time, I'm not the creator nor maintainer): https://rentry.co/GPT-SoVITS-guide#/

### Unofficial Guide (Windows Only):
Mostly the same info as in the official.

Install the requirements
```
pip install -r requirements.txt
```
I needed to change `numba==0.56.4` to just `numba`.

Place the `GPT-SoVITS-beta0217` into the voiceAI directory and then navigate inside of `GPT-SoVITS-beta0217` and run the `go-webui.bat`. Finally, navigate to the given HTTP address on a web browser.

#### Dataset Creation
This can be done either using the GUI `0-Fetch dataset`, but I used Davinci Resolve to more precisely cut and export the audio in .wav format (.wav format is not necessary, but it's what I use).

#### Preprocessing Audio
(Following is semi-optional, mainly worth it if you haven't already preprocessed the audio yourself)
Next, I recommend checking the `Open UVR5-WebUI` box, and in this new page providing the folder in which you have all of the individual .wav files ready. For the `Model` using `HP2` or `HP5`, you can pretty effectively isolate the speaker voice from the background noise/music. It's also good to select the same audio format to export to. You can also check out the other ones, I've found so far that I only need to isolate the speaker audio from the background noise/music. 

#### Audio Labelling
In `0-Fetch dataset` tab, using the `ASR` tool, `Faster Whisper` and language set to `en` (or zh/ja if using those), this will create surprisingly good results (generally only need to do minor corrections).

## Training
First in the `1A-Dataset formatting` tab, give the model a name.

IMPORTANT: You must only provide the that to the .list file for the `Text labelling file` (ex. `/path/to/test.list`). BUT, you musn't provide the path to the `Audio dataset folder`, otherwise during training you can/will run into `division by zero error`. There's no clear info as to why, I suspect it has to do with the fact that the .list file already contains the paths to the audio files.

After this, in the `1B-Fine-tuned training` tab, train both the SoVITS and GPT. So far, I've left everything to default.


## Inference
The GUI is good for basic testing, but since we're integrating this to an LLM it's not going to work.

(Optional, only in case you want to do something specific):
- Navigate to the `fast_inference_` branch: https://github.com/RVC-Boss/GPT-SoVITS/tree/fast_inference_
- Download the zip
- Copy the `ffmpeg` and `ffprobe` files from `GPT-SoVITS-beta0217`
- navigate to `GPT-SoVITS-fast_inference_\GPT_SoVITS\pretrained_models` add the following files/folders (can be found in equivalent folder in `GPT-SoVITS-beta0217`)

```
s2G488k.pth
s2D488k.pth
chinese-hubert-base
chinese-roberta-wwm-ext-large
s1bert25hz-2kh-longer-epoch=68e-step=50232
```
You will also need to install `wordsegment` if not already installed
```
pip install wordsegment
```

### Running Inference
> Inside of `GPT-SoVITS-fast_inference`

In the CMD run:
```
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

:information_source: Weights only need to be set once, also done through browser

:heavy_exclamation_mark: This section is WIP, model weights will be uploaded to huggingface at a later date. For now, ignore the setting weights section

Navigate to the local addresses below in a browser (or send requests through CMD)

set the GPT weights:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/custom_model/JohnSmith0-e15.ckpt
```

Set the SoVITS weights:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/custom_model/JohnSmith0_e8_s64.pth
```

:information_source: While this is fine to do, I recommend using TTS.py or brain.py
Do inference:
```
http://127.0.0.1:9880/tts?text=But truly, is a simple piece of paper worth the credit people give it?&text_lang=en&ref_audio_path=../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav&prompt_lang=en&prompt_text=But truly, is a simple piece of paper worth the credit people give it?&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true&top_k=5&top_p=1&temperature=1
```

# Acknowledgements
> I used a mix of my own and other people's code for the LLM training and evaluation. I can't remember who the people are, sorry. I would cite otherwise.

This project uses the following tools:

[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/tree/main)