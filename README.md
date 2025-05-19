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
    * [Fine-tuning](#training-fine-tuning)
    * [Quantization](#training-fine-tuning)
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
* [Testing](#testing)
* [Acknowledgements](#acknowledgements)

# Features
* Fine-tune an LLM
* Train and use a custom voice model for TTS (see acknowledgements)
* Speak to your LLM using STT (see acknowledgements)

  **Features:**
  - [X] Create a function to create a dataset for LLM training from a .csv file
  - [ ] Send audio data to Discord or other, so anyone in call can hear
  - [X] Support for receiving and responding to YouTube live chat messages
  - [X] Support for receiving and responding to Twitch live chat messages
  - [ ] Support for receiving and responding to Kick live chat messages
  - [ ] Boolean for automatically stopping the model from speaking when user speaks at the same time
  - [ ] On screen subtitles for OBS
  - [ ] As LLM generates text, receive it and use TTS to speak it aloud for potentially better real-time conversation
  - [ ] Vision capabilities, see what is on screen and commentate on it
  - [ ] Gaming capabilities, play different types of games
  - [ ] Vtuber model capabilities, movement, expressions and lipsync, etc
  - [ ] RAG for enhanced conversational cohesion
  - [ ] Recognition of separate speakers
  - [ ] run LLM and TTS on separate threads (benefits?)

  **Unsure Features:**
  - [ ] Drawing capability?
  - [ ] Singing capability?
  - [ ] Vector database for improved RAG?
  - [ ] Different/Custom STT?

# Setup
This is developed and tested on Python 3.12.3.

## installation

:exclamation: This mainly works on Linux (Ubuntu 24.04 LTS, other distros may work), Windows support will be dropped soon.

[flash-attention](https://github.com/Dao-AILab/flash-attention) is required (used by ExllamaV2).

PyTorch (assumes you are using CudaToolkit 12.4)
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

```

TensorboardX (PyTorch should install this automatically):
```
pip install tensorboardX
```


[ExllamaV2](https://github.com/turboderp/exllamav2) 
:information_source: Assumes Pytorch 2.6.0, CudaToolkit 12.4, Python 3.12

```
pip install https://github.com/turboderp-org/exllamav2/releases/download/v0.2.9/exllamav2-0.2.9+cu124.torch2.6.0-cp312-cp312-linux_x86_64.whl
```

The local version of Whisper needs to be installed manually
```
pip install SpeechRecognition[whisper-local]
```

may need to use:
```
python -m nltk.downloader averaged_perceptron_tagger_eng
```

Rest of the requirements can be installed via:
```
pip install -r requirements.txt
```

## Virtual Environments
The virtual environment simply helps to avoid package conflicts. Do note that this will take more space in the storage as each environment is its own.

Create env (the last `venv` is the folder name/path where the venv will be created):

```
 python -m venv venv
```

Activate env:
```
venv/bin/activate
```

Deactivate env:
```
deactivate
```

# Quick Start
:exclamation: Heavy WIP

After activating the venv, run the following in the root directory:
```
python run.py
```

:information_source: The next sub-sections are purely optional, only necessary if you want the AI to interact with livechat on YouTube/Twitch/Kick.

## .env File Setup (Optional)
Create a **.env** file inside of the root directory and add as much of the following as desired:

:information_source: For information on YouTube/Twitch related variables refer to their respective sections: [YouTube API](#youtube-api) [Twitch API](#twitch-api).

```
#For all
CONVERSATION_LOG_FILE=livechatAPI/data/llm_finetune_data.csv

#For YouTube
YT_FETCH=False
YT_API_KEY=
YT_CHANNEL_ID=
LAST_NEXT_PAGE_TOKEN=

#For Twitch
TW_FETCH=False
TW_CHANNEL=
TW_BOT_NICK=Botty
TW_CLIENT_ID=
TW_CLIENT_SECRET=
TW_ACCESS_TOKEN=
TW_USE_THIRD_PARTY_TOKEN=False #True if using non-locally token
```
<details>
<summary>Variable Explanations</summary>

* CONVERSATION_LOG_FILE: The .csv file which the user message AND the LLM response is written to

**YouTube**:
* YT_FETCH: (Boolean) True if fetching youtube live chat messages
* YT_API_KEY: Your YouTube project API key
* YT_CHANNEL_ID: Your YouTube channel id (NOT channel name)
* LAST_NEXT_PAGE_TOKEN: Last fetched messages from live chat -- HANDLED BY PROGRAM DO NOT TOUCH

**TWITCH**:
* TW_FETCH: (Boolean) True if fetching twitch live chat messages
* TW_CHANNEL: Your Twitch channel name -- whatever you named your channel
* TW_BOT_NICK: Default name is 'Botty', can be renamed to anything
* TW_CLIENT_ID: Your App's client id
* TW_CLIENT_ID: Your App's client secret
* TW_ACCESS_TOKEN: If generated LOCALLY -> technically refresh access token, if using third party token -> is actually the access token
* TW_USE_THIRD_PARTY_TOKEN: (Boolean) True if using a token NOT generated locally ex. https://twitchtokengenerator.com/

</details>


## YouTube API (Optional)
The official guide here: https://developers.google.com/youtube/v3/live/getting-started

Basic steps:
1. Navigate to the [Google API Console](https://console.cloud.google.com/apis/credentials)
2. Create a new project
3. Back at the credentials page, use the **CREATE CREDENTIALS** to generate an API key
4. Paste the API key in the **.env** file in the **YT_API_KEY**
5. Go to Settings (click user icon top right)
6. Go to Advanced settings and copy the **Channel ID** to **YT_CHANNEL_ID** in **.env** file

The environment variable **LAST_NEXT_PAGE_TOKEN** is handled by the program itself (DO NOT TOUCH). It makes sure that upon program restart, we continue to fetch live chat messages from where we last left off.

## Twitch API (Optional)
:information_source: The twitchio bot automatically renews your token using the your twitch application client id and client secret

Precursory steps:
1. Set **TW_CHANNEL=<your twitch username>** and **TW_BOT_NICK=<any name for the bot>**
2. You will first need to create an application. Navigate to the developer console: https://dev.twitch.tv/console/
3. Click on **Register Your Application** -- right side
4. Set the following:
**Name**: the application name can be anything
**OAuth Redirect URLs**: you can use your desired HTTPS URI, in Optional Path 1 the following is used:
```
https://localhost:8080
```
**Category**: Chat Bot -- could use others probably
**Client Type**: Confidential -- could be Public if need be
5. Back at the console page, click on **Manage**
6. Copy the **Client ID** and **Client Secret** to the **.env** file's variables **TW_CLIENT_ID=<Client ID here>** and **TW_CLIENT_SECRET=<Client Secret here>**

Optional Path 1:
Basic steps:
1. Run `run.py` OR `twitch.py` (may need to uncomment some code) -- assumes you have all the requirements installed
2. A locally hosted HTTPS web server should be running and a web page should open on your default browser -- MOST LIKELY THE BROWSER WILL WARN ABOUT THE CONNECTION, JUST ALLOW IT... or don't (see optional path 2)
3. Authorize yourself to generate your own tokens
4. You're done! For the foreseeable future, the refresh token will handle everything automatically, no need for steps 2 and 3

Optional Path 2:
1. Navigate to and generate a token: https://twitchtokengenerator.com/
2. Set **TW_THIRD_PARTY_TOKEN=** to `True` -- case sensitive
3. Done!

In Optional Path 1, the TW_ACCESS_TOKEN is technically the refresh token, only self.TOKEN in TwitchAuth class is an actual access token. This is ONLY a technicality and does not affect anything.

The twitchio bot should presumably automatically renew your token upon expiration. This requires atleast **Client Secret** and maybe **Client ID** -- UNTESTED.

# Large Language Model (LLM)
>:information_source: HEAVY WIP
## Prompt Style


```
<|im_start|>system <character behaviour> <|im_end|> <|im_start|>context <information about current situation (previous dialogue or description of situation)> <|im_end|> <|im_start|>user <Question or Instruction> <|im_end|> <|im_start|>assistant <Model Answer>
```

## Dataset preparation
For the WIP dataset creator, the base dataset will be expected to be in a .csv file format


>:information_source: The following are examples for the dataset formatting (will be the end result of dataset creator later on as well)
```
<|im_start|>system assistant is a cocky snarky bastard<|im_end|><|im_start|>context assistant is at a coffee shop deciding which coffee to order.<|im_end|><|im_start|>user what are you doing?<|im_end|> <|im_start|>assistant Edging hard<|im_end|>
```

```
<|im_start|>system assistant is a cocky snarky bastard<|im_end|><|im_start|>context assistant is at a coffee shop deciding which coffee to order. user: what are you doing here?<|im_end|><|im_start|>user<|im_end|> <|im_start|>assistant Hmm, perhaps a coffee<|im_end|>
```
Effectively a **System-Context-User-Assistant** format is being followed (**SCUA** referred to as **SICUEAC** in research.pdf [WIP]).

## Fine-tuning
> :warning: HEAVY WIP, also fine-tuned model != quantized model

In the `finetune.py` file, only the **DATASET_PATH** must be changed:
- **BASE_MODEL**: the base model that is to be fine-tuned -- currently uses [NousResearch/Hermes-2-Theta-Llama-3-8B](#acknowledgements)
- **NEW_MODEL**: the fine-tuned model's name -- technically the output directory for it
- **OUTPUT_DIR**: fine-tuning process' output directory
- **DATASET_PATH**: path to the dataset file -- currently .txt, soon .parquet

## Quantization


:information_source: Assumes ExllamaV2 was installed via the wheel -- Works but still WIP guide

```
python -m exllamav2.conversion.convert_exl2
```

## Inference
:exclamation: HEAVY WIP

text_gen_test.py currently works as the inference script.

## Huggingface Cache Cleaning

Install
```
pip install huggingface_hub["cli"]
```

run
```
huggingface-cli delete-cache
```

Next, select model(s) to delete from cache.

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

IMPORTANT: You must only provide the path to the `.list` file for the `Text labelling file` field (ex. `/path/to/test.list`). BUT, you musn't provide the path to the `Audio dataset folder`, otherwise during training you can/will run into `division by zero error`. There's no clear info as to why, I suspect it has to do with the fact that the .list file already contains the paths to the audio files.

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
You will also need to install `wordsegment` if not already installed (should be installed when using requirements.txt)
```
pip install wordsegment
```

### Running Inference
> Inside of `GPT-SoVITS-fast_inference`

In the CMD run:
```
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

:information_source: Weights only need to be set once, also this is done through the browser, CMD is also possible

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

Do basic inference through browser:
```
http://127.0.0.1:9880/tts?text=But truly, is a simple piece of paper worth the credit people give it?&text_lang=en&ref_audio_path=../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav&prompt_lang=en&prompt_text=But truly, is a simple piece of paper worth the credit people give it?&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true&top_k=7&top_p=0.87&temperature=0.87
```

# Testing
:warning: Running just **pytest** will result in running all test files in the entire project which will inevitably fail. We are only concerned with the tests within the tests directory at the root.

To run all tests use:
```
pytest -s tests
```

Only unit tests:
```
pytest -s tests -m "not integration"
```

Only integration tests
```
pytest -s tests -m integration
```

# Acknowledgements
This project makes use of the following:

* [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/tree/main)
* [CapybaraHermes](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ)
* [NousResearch/Hermes-2-Theta-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B/tree/main)
* [Unsloth](https://github.com/unslothai/unsloth)
* [Speech_Recognition](https://github.com/Uberi/speech_recognition)
* [Dao-AILab](https://github.com/Dao-AILab/flash-attention)
* [turboderp](https://github.com/turboderp/exllamav2)