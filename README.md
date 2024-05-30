# Vtuber-AI
The aim of this project is both to give a good starting point for anyone to create their own custom AI Vtubers (technically won't have to be a Vtuber), and be something fun for me to build and potentially have my work live on and be incorporated into numerous other projects for a long time to come.

## Table of Contents

* [Vtuber-AI](#vtuber-ai)
    * [Features](#features)
    * [Todo List](#todo-list)
* [Setup](#setup)
    * [Virtual Environments](#virtual-environments)
* [Large Language Model (LLM)](#large-language-model-llm)
    * [Prompt Style](#prompt-style)
    * [Dataset preparation](#dataset-preparation)
    * [Training (Fine-tuning)](#training-fine-tuning)
    * [Inference](#inference)
* [Voice Model](#voice-model)
    * [Training](#training)
        * [Official Guide](#official-guide)
        * [Unofficial Guide (Windows Only)](#unofficial-guide-windows-only)
            * [Dataset Creation](#dataset-creation)
            * [Preprocessing Audio](#preprocessing-audio)
            * [Audio Labelling](#audio-labelling)
    * [Training](#training-1)
    * [Inference](#inference-1)
* [Acknowledgements](#acknowledgements)

# Features
* Fine-tune an LLM
* Train and use a custom voice model for TTS (see acknowledgements)
* Speak to your LLM using STT (see acknowledgements)

## Todo List
  **Misc:**
  - [X] Refractor files, move LLM related files to its own directory
  - [X] Create a proper requirements.txt
  - [ ] Create a function to create a dataset for LLM training from a .csv file
  - [X] modify train.py to either not use list of model names or to make it a separate function
  - [ ] Improve README guide
  - [ ] Write research.pdf
  - [ ] Change character sheet to be a .json type of file
  - [ ] Change LLM inference question sheet to be a .json type of file
  - [ ] Create and publish proper custom LLM model to HuggingFace
  - [ ] Create and publish proper custom voice model somewhere

  **Features:**
  - [ ] Send audio data to Discord or other, so anyone in call can hear
  - [ ] Support for receiving and responding to YouTube live chat messages
  - [ ] Support for receiving and responding to Twitch live chat messages
  - [ ] Support for receiving and responding to Kick live chat messages
  - [ ] Boolean for automatically stopping the model from speaking when user speaks at the same time
  - [ ] On screen subtitles for OBS
  - [ ] As LLM generates text, receive it and use TTS to speak it aloud for potentially better real-time conversation
  - [ ] Vision capabilities, see what is on screen and commentate on it
  - [ ] Gaming capabilities, play different types of games
  - [ ] Vtuber model capabilities, movement, expressions and lipsync, etc
  - [ ] RAG for enhanced conversational cohesion
  - [ ] Recognition of separate speakers

  **Unsure Features:**
  - [ ] Drawing capability?
  - [ ] Singing capability?
  - [ ] Vector database for improved RAG?
  - [ ] Different/Custom STT?
# Setup
This is developed and tested on Python 3.11.8.

In the root directory, install everything in the requirements.txt. All of the necessary packages are included in this file, no need to install anything anywhere else. (I highly suggest setting up a virtual environment up first).
```
pip install -r requirements.txt
```

## Virtual Environments
The virtual environment simply helps to avoid package conflicts. Do note that this will take more space in the storage as each environment is its own.

:information_source: Note that this is for CMD

Create env (the last `venv` is the folder name/path where the venv will be created):
```
 python -m venv venv
```

Activate env:
```
venv\Scripts\activate
```

Deactivate env:
```
 deactivate
```
Then just delete the venv folder

# Large Language Model (LLM)
## Prompt Style

>:information_source: This is the prompt styling for inference

```
<|im_start|>system <character behaviour> <|im_end|> <|im_start|>context <information about current situation (previous dialogue or description of situation)> <|im_end|> <|im_start|>user <Question or Instruction> <|im_end|> <|im_start|>assistant <Model Answer>
```

## Dataset preparation
For the WIP dataset creator, the dataset will be expected to be in a .csv file format


>:information_source: The following are examples for the dataset formatting (will be the end result of dataset creator later on as well)
```
<|im_start|>system assistant is a cocky snarky bastard<|im_end|><|im_start|>context assistant is at a coffee shop deciding which coffee to order.<|im_end|><|im_start|>user what are you doing?<|im_end|> <|im_start|>assistant Edging hard<|im_end|>
```

```
<|im_start|>system assistant is a cocky snarky bastard<|im_end|><|im_start|>context assistant is at a coffee shop deciding which coffee to order. user: what are you doing here?<|im_end|><|im_start|>user<|im_end|> <|im_start|>assistant Hmm, perhaps a coffee<|im_end|>
```
Effectively a **System-Context-User-Assistant** format is being followed (**SCUA** referred to as **SICUEAC** in research.pdf [WIP]).

## Training (Fine-tuning)
> :warning: It is assumed that the model name and .txt file (containing the data in expected format) have the exact same names.
> Example: Model name = johnsmith, .txt = johnsmith.txt

*Fine-tuning* is the accurate term, however, I believe *training* is more universally understood.

In the `train.py` file, for remove the other names in **model_names** variable and add your model name. If need be, the base model can be changed in the **model_preprocess** function, inside of the  **model_name** variable.

Now, running `train.py` should train the model. I'm unsure if this works incase the base model is not a **quantized** model. Also, depending on your hardware, you may need to change some of the hyper parameters. 

## Inference
:exclamation: HEAVY WIP

eval.py with modifications can be used for the inference, this is planned to change very soon.

# Voice Model
## Training
### Official Guide: 
follow the user guide from: https://github.com/RVC-Boss/GPT-SoVITS/tree/main

user guide direct link (can change at any time, I'm not the creator nor maintainer): https://rentry.co/GPT-SoVITS-guide#/

### Unofficial Guide (Windows Only):
Mostly the same info as in the official.

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

This project makes use of the following projects:

* [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/tree/main)
* [CapybaraHermes](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ)
* [Speech_Recognition](https://github.com/Uberi/speech_recognition)