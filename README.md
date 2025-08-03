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

:exclamation: This mainly works on Linux (Ubuntu 24.04 LTS, other distros may work). Direct Windows support has been dropped -- only works through WSL2 OR available in the Windows-legacy branch (heavily out-of-date but technically functional-ish).

[flash-attention](https://github.com/Dao-AILab/flash-attention) is required (used by ExllamaV2).

PyTorch 2.7.1 (assumes you are using CudaToolkit 12.8)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

[ExllamaV2](https://github.com/turboderp/exllamav2) 
:information_source: Assumes Pytorch 2.7.1, CudaToolkit 12.8, Python 3.12

```
https://github.com/turboderp-org/exllamav2/releases/download/v0.3.1/exllamav2-0.3.1+cu128.torch2.7.0-cp312-cp312-linux_x86_64.whl
```


may need to install -- unlikely, haven't had to do this in recent venv builds:
```
python -m nltk.downloader averaged_perceptron_tagger_eng
```

## Virtual Environments
The virtual environment simply helps to avoid package conflicts. Do note that this will take more space in the storage as each environment is its own.

Create env (the last `venv` is the folder name/path where the venv will be created):

```
 python3 -m venv venvRun
```

Activate env:
```
source venv/bin/activate
```

Deactivate env:
```
deactivate
```

# Quick Start
:exclamation: Heavy WIP

After activating the venv, run the following in the root directory:
```
python -m run
```

:information_source: The next sub-sections are purely optional, only necessary if you want the AI to interact with livechat on YouTube/Twitch/Kick.

## .env File Setup (Optional)
Create a **.env** file inside of the root directory and add as much of the following as desired:

:information_source: For information on YouTube/Twitch related variables refer to their respective sections: [YouTube API](#youtube-api) [Twitch API](#twitch-api).

```
#For all
CONVERSATION_LOG_FILE=path/to/conversation_log.csv
HIGH_CHAT_VOLUME=False #Many viewers typing into chat(s)

#For YouTube
YT_FETCH=False
YT_OAUTH2_JSON=your_client_secret.json
YT_API_KEY=YourAPIKEY1234
YT_CHANNEL_ID=YourYTChannelID
LAST_NEXT_PAGE_TOKEN=HandledAutomatically

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
5. Go through the **OAuth consent screen** and setup OAuth for your project.
6. Back at the credentials page, download the **JSON** file of your OAuth client.
7. Go to Settings (click user icon top right)
8. Go to Advanced settings and copy the **Channel ID** to **YT_CHANNEL_ID** in **.env** file

The environment variable **LAST_NEXT_PAGE_TOKEN** is handled by the program itself (DO NOT TOUCH). It makes sure that upon program restart, the program continues to fetch live chat messages from where it last left off.

## Twitch API (Optional)
:information_source: The twitchio bot automatically renews your token using the your twitch application client id and client secret


:information_source: Once step 3 `Run and start the bot from the code below.` is reached in the guide, run the `twitch.py` file in `Livechat_Wizard`. This will automatically handle setting up the **TW_OWNER_ID** and **TW_BOT_ID** for the `.env`. Once running, continue with the guide.

The official guide: https://twitchio.dev/en/stable/getting-started/quickstart.html

# Large Language Model (LLM) Fine-tuning
>:information_source: HEAVY WIP

The modularity of the LLM system allows for any LLM that is in some way compatible with ExLlamaV2 to be used -- it's also possible to extend the models.py class for something different. The **dialogue_service.py** itself doesn't care too much about what is used, only that an async streamable generator is provided for consumption.

## Fine-tuning
> :warning: HEAVY WIP, also fine-tuned model != quantized model
[Unsloth](https://github.com/unslothai/unsloth) is used for the fine-tuning portion.

### Dataset preparation

#### CSV file requirements
The **.csv** is expected to have the columns as follows:
`user, character, context, conversation_id`

**user**: This is the actual user prompt -- e.g. Do you edge?
**character**: This is the LLM response -- e.g. Yes, yes I do indubitably.
**context**: Any context information -- e.g. Edging is a highly competitive sport OR John: What should we talk about?
**conversation_id**: An **int** value which is used to build the full longer conversations for the dataset -- assumes each row with the same ID is part of the same conversation

#### CSV to PARQUET creation
In **dataset_utils.py**, the following constants need to be set to your values:
**BASE_PATH**: The base path to where your dataset will reside in
**CSV_PATH**: BASE_PATH + <your actual .csv file name>
**MODEL_PATH**: This is the to be fine-tuned base model path (huggingface or local directory) -- used for applying correct chat template
**CHARACTER_JSON_PATH**: JSON file that contains `instructions`
**MAX_TURNS**: How many turns to keep in memory -- keep to the same value as what would be used in practice

From the dataset, 3 parquet files are made:
1. Clean one that just saves the CSV to Parquet
2. Fine-tuning parquet dataset with the auto chat template pre-applied
3. Quantization calibration dataset for ExlLamaV2

### Unsloth based fine-tuning
In the `finetune.py` file, the following constants need to be set to your desired model's values:
**FINETUNING_MODE**: `language` or `vision` -- vision assumes your model is vision capable AND that your dataset is structured correctly for it
**MODEL_ID**: This is the to be fine-tuned model -- usually use an unsloth's available models if possible
**DATASET_PATH**: Path to your fine-tuning dataset.parquet file
**OUTPUT_DIR**: Where to save temp files during fine-tuning
**SAVE_DIR**: Where to save final merged fine-tuned model to

## Quantization
:information_source: Assumes ExllamaV2 was installed via the wheel -- Works but still WIP guide

```


python -m exllamav2.conversion.convert_exl2 -i LLM_Wizard/qwen2.5-vl-finetune-merged -o LLM_Wizard/exl2_out -b 4.0 -hb 6 -c LLM_Wizard/dataset/exllama_calibration.parquet -l 2040 -r 100 -ss 0
```

The **mr** and possibly the **ml** flags need to be set differently to fit be able to fit your data.
```
python -m exllamav2.conversion.convert_exl2 -i LLM_Wizard/qwen2.5-vl-finetune-merged -o LLM_Wizard/exl2_out -c LLM_Wizard/dataset/exllama_calibration.parquet -nr -l 100 -r 10 -mr 3 -b 8.0

```

## Inference
:exclamation: FOR EXLLAMAV2, YOU MAY NEED TO EDIT THE MODEL CONFIG.JSON AND REMOVE THE OTHER ROPE_SCALING TYPES AND ONLY KEEP THE MROPE

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


# Voice Model -- Text-To-Speech (TTS)
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
* [Unsloth](https://github.com/unslothai/unsloth)
* [Dao-AILab](https://github.com/Dao-AILab/flash-attention)
* [turboderp](https://github.com/turboderp/exllamav2)
* [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
