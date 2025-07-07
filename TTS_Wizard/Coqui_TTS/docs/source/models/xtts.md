# XTTS
XTTS is a super cool Text-to-Speech model that lets you clone voices in different languages by using just a quick 3-second audio clip. Built on the 🐢Tortoise,
XTTS has important model changes that make cross-language voice cloning and multi-lingual speech generation super easy.
There is no need for an excessive amount of training data that spans countless hours.

## Features
- Voice cloning.
- Cross-language voice cloning.
- Multi-lingual speech generation.
- 24khz sampling rate.
- Streaming inference with < 200ms latency. (See [Streaming inference](#streaming-manually))
- Fine-tuning support. (See [Training](#training))

## Updates with v2
- Improved voice cloning.
- Voices can be cloned with a single audio file or multiple audio files, without any effect on the runtime.
- Across the board quality improvements.

## Code
Current implementation only supports inference and GPT encoder training.

## Languages
XTTS-v2 supports 17 languages:

- Arabic (ar)
- Chinese (zh-cn)
- Czech (cs)
- Dutch (nl)
- English (en)
- French (fr)
- German (de)
- Hindi (hi)
- Hungarian (hu)
- Italian (it)
- Japanese (ja)
- Korean (ko)
- Polish (pl)
- Portuguese (pt)
- Russian (ru)
- Spanish (es)
- Turkish (tr)

## License
This model is licensed under [Coqui Public Model License](https://coqui.ai/cpml).

## Contact
Come and join in our 🐸Community. We're active on [Discord](https://discord.gg/fBC58unbKE) and [Github](https://github.com/idiap/coqui-ai-TTS/discussions).

## Inference

### 🐸TTS Command line

You can check all supported languages with the following command:

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --list_language_idx
```

You can check all Coqui available speakers with the following command:

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --list_speaker_idx
```

#### Coqui speakers
You can do inference using one of the available speakers using the following command:

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent." \
     --speaker_idx "Ana Florence" \
     --language_idx en \
     --use_cuda
```

#### Clone a voice
You can clone a speaker voice using a single or multiple references:

##### Single reference

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "Bugün okula gitmek istemiyorum." \
     --speaker_wav /path/to/target/speaker.wav \
     --language_idx tr \
     --use_cuda
```

##### Multiple references
```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "Bugün okula gitmek istemiyorum." \
     --speaker_wav /path/to/target/speaker.wav /path/to/target/speaker_2.wav /path/to/target/speaker_3.wav \
     --language_idx tr \
     --use_cuda
```
or for all wav files in a directory you can use:

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "Bugün okula gitmek istemiyorum." \
     --speaker_wav /path/to/target/*.wav \
     --language_idx tr \
     --use_cuda
```

### 🐸TTS API

#### Clone a voice
You can clone a speaker voice using a single or multiple references:

##### Single reference

Splits the text into sentences and generates audio for each sentence. The audio files are then concatenated to produce the final audio.
You can optionally disable sentence splitting for better coherence but more VRAM and possibly hitting models context length limit.

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav=["/path/to/target/speaker.wav"],
                language="en",
                split_sentences=True
                )
```

##### Multiple references

You can pass multiple audio files to the `speaker_wav` argument for better voice cloning.

```python
from TTS.api import TTS

# using the default version set in 🐸TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# using a specific version
# 👀 see the branch names for versions on https://huggingface.co/coqui/XTTS-v2/tree/main
# ❗some versions might be incompatible with the API
tts = TTS("xtts_v2.0.2").to("cuda")

# getting the latest XTTS_v2
tts = TTS("xtts").to("cuda")

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav=["/path/to/target/speaker.wav", "/path/to/target/speaker_2.wav", "/path/to/target/speaker_3.wav"],
                language="en")
```

#### Coqui speakers

You can do inference using one of the available speakers using the following code:

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# generate speech by cloning a voice using default settings
tts.tts_to_file(
  text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
  file_path="output.wav",
  speaker="Ana Florence",
  language="en",
  split_sentences=True
)
```


### 🐸TTS Model API

To use the model API, you need to download the model files and pass config and model file paths manually.

### Manual Inference

If you want to be able to `load_checkpoint` with `use_deepspeed=True` and **enjoy the speedup**, you need to install deepspeed first.

```console
pip install deepspeed
```

#### Inference parameters

- `text`: The text to be synthesized.
- `language`: The language of the text to be synthesized.
- `gpt_cond_latent`: The latent vector you get with get_conditioning_latents. (You can cache for faster inference with same speaker)
- `speaker_embedding`: The speaker embedding you get with get_conditioning_latents. (You can cache for faster inference with same speaker)
- `temperature`: The softmax temperature of the autoregressive model. Defaults to 0.65.
- `length_penalty`: A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs. Defaults to 1.0.
- `repetition_penalty`: A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.
- `top_k`: Lower values mean the decoder produces more "likely" (aka boring) outputs. Defaults to 50.
- `top_p`: Lower values mean the decoder produces more "likely" (aka boring) outputs. Defaults to 0.8.
- `speed`: The speed rate of the generated audio. Defaults to 1.0. (can produce artifacts if far from 1.0)
- `enable_text_splitting`: Whether to split the text into sentences and generate audio for each sentence. It allows you to have infinite input length but might loose important context between sentences. Defaults to False.


#### Inference


```python
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference.wav"])

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
```

You can also use the Coqui speakers:

```python
gpt_cond_latent, speaker_embedding = model.speaker_manager.speakers["Ana Florence"].values()
```

#### Streaming manually

Here the goal is to stream the audio as it is being generated. This is useful for real-time applications.
Streaming inference is typically slower than regular inference, but it allows to get a first chunk of audio faster.


```python
import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference.wav"])

print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding
)

wav_chuncks = []
for i, chunk in enumerate(chunks):
    if i == 0:
        print(f"Time to first chunck: {time.time() - t0}")
    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    wav_chuncks.append(chunk)
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
```


## Training

### Easy training
To make `XTTS_v2` GPT encoder training easier for beginner users we did a gradio demo that implements the whole fine-tuning pipeline. The gradio demo enables the user to easily do the following steps:

- Preprocessing of the uploaded audio or audio files in 🐸 TTS coqui formatter
- Train the XTTS GPT encoder with the processed data
- Inference support using the fine-tuned model

The user can run this gradio demo locally or remotely using a Colab Notebook.

#### Run demo on Colab
To make the `XTTS_v2` fine-tuning more accessible for users that do not have good GPUs available we did a Google Colab Notebook.

The Colab Notebook is available [here](https://colab.research.google.com/github/idiap/coqui-ai-TTS/blob/dev/TTS/demos/xtts_ft_demo/XTTS_finetune_colab.ipynb).

To learn how to use this Colab Notebook please check the [XTTS fine-tuning video](https://www.youtube.com/watch?v=8tpDiiouGxc).

If you are not able to acess the video you need to follow the steps:

1. Open the Colab notebook and start the demo by runining the first two cells (ignore pip install errors in the first one).
2. Click on the link "Running on public URL:" on the second cell output.
3. On the first Tab (1 - Data processing) you need to select the audio file or files, wait for upload, and then click on the button "Step 1 - Create dataset" and then wait until the dataset processing is done.
4. Soon as the dataset processing is done you need to go to the second Tab (2 - Fine-tuning XTTS Encoder) and press the button "Step 2 - Run the training" and then wait until the training is finished. Note that it can take up to 40 minutes.
5. Soon the training is done you can go to the third Tab (3 - Inference) and then click on the button "Step 3 - Load Fine-tuned XTTS model" and wait until the fine-tuned model is loaded. Then you can do the inference on the model by clicking on the button "Step 4 - Inference".


#### Run demo locally

To run the demo locally you need to do the following steps:
1. Install   🐸 TTS following the instructions available [here](https://coqui-tts.readthedocs.io/en/latest/installation.html).
2. Install the Gradio demo requirements with the command `python3 -m pip install -r TTS/demos/xtts_ft_demo/requirements.txt`
3. Run the Gradio demo using the command `python3 TTS/demos/xtts_ft_demo/xtts_demo.py`
4. Follow the steps presented in the [tutorial video](https://www.youtube.com/watch?v=8tpDiiouGxc&feature=youtu.be) to be able to fine-tune and test the fine-tuned model.


If you are not able to access the video, here is what you need to do:

1. On the first Tab (1 - Data processing) select the audio file or files, wait for upload
2. Click on the button "Step 1 - Create dataset" and then wait until the dataset processing is done.
3. Go to the second Tab (2 - Fine-tuning XTTS Encoder) and press the button "Step 2 - Run the training" and then wait until the training is finished. it will take some time.
4. Go to the third Tab (3 - Inference) and then click on the button "Step 3 - Load Fine-tuned XTTS model" and wait until the fine-tuned model is loaded.
5. Now you can run inference with the model by clicking on the button "Step 4 - Inference".

### Advanced training

A recipe for `XTTS_v2` GPT encoder training using `LJSpeech` dataset is available at https://github.com/coqui-ai/TTS/tree/dev/recipes/ljspeech/xtts_v1/train_gpt_xtts.py

You need to change the fields of the `BaseDatasetConfig` to match your dataset and then update `GPTArgs` and `GPTTrainerConfig` fields as you need. By default, it will use the same parameters that XTTS v1.1 model was trained with. To speed up the model convergence, as default, it will also download the XTTS v1.1 checkpoint and load it.

After training you can do inference following the code bellow.

```python
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Add here the xtts_config path
CONFIG_PATH = "recipes/ljspeech/xtts_v1/run/training/GPT_XTTS_LJSpeech_FT-October-23-2023_10+36AM-653f2e75/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "recipes/ljspeech/xtts_v1/run/training/XTTS_v2_original_model_files/vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "recipes/ljspeech/xtts_v1/run/training/GPT_XTTS_LJSpeech_FT/best_model.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = "LjSpeech_reference.wav"

# output wav path
OUTPUT_WAV_PATH = "xtts-ft.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)
```



## References and Acknowledgements
- VallE: https://arxiv.org/abs/2301.02111
- Tortoise Repo: https://github.com/neonbjb/tortoise-tts
- Faster implementation: https://github.com/152334H/tortoise-tts-fast
- Univnet: https://arxiv.org/abs/2106.07889
- Latent Diffusion:https://arxiv.org/abs/2112.10752
- DALL-E: https://arxiv.org/abs/2102.12092
- Perceiver: https://arxiv.org/abs/2103.03206


## XttsConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.xtts_config.XttsConfig
    :members:
```

## XttsArgs
```{eval-rst}
.. autoclass:: TTS.tts.models.xtts.XttsArgs
    :members:
```

## XTTS Model
```{eval-rst}
.. autoclass:: TTS.tts.models.xtts.Xtts
    :members:
```
