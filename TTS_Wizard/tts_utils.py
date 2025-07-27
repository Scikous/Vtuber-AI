import logging

def prepare_tts_params_gpt_sovits(text_to_speak, text_lang="en", ref_audio_path="../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav", prompt_text="", prompt_lang="en", streaming_mode=False, media_type="wav", logger=None):
    """
    Prepares the dictionary of parameters for the GPT_SoVITS TTS service.
    All parameters are now passed directly to the function.
    """    
    return {
        "text": text_to_speak,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "streaming_mode": streaming_mode,
        "media_type": media_type,
    }
def prepare_tts_params_xtts(text_to_speak, text_lang="en", speech_speed=1.0, streaming_mode=True):
    """
    Prepares the dictionary of parameters for the Coqui-AI-TTS XTTS TTS service.
    All parameters are now passed directly to the function.
    """    
    return {
        "text": text_to_speak,
        "language": text_lang,
        "speech_speed": speech_speed,
        "streaming_mode": streaming_mode,
    }

#refactor for RealtimeTTS
def prepare_tts_params_rtts(text_to_speak, text_lang="en", min_sentence_len=8, speech_speed=1.0, streaming_mode=True):
    """
    Prepares the dictionary of parameters for the Coqui-AI-TTS XTTS TTS service.
    All parameters are now passed directly to the function.
    """    
    return {
        "text": text_to_speak,
        "language": text_lang,
        "min_sentence_len": min_sentence_len,
        "speech_speed": speech_speed,
        "streaming_mode": streaming_mode,
    }
