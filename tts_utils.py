import logging

def prepare_tts_params(text_to_speak, text_lang="en", ref_audio_path="../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav", prompt_text="", prompt_lang="en", streaming_mode=False, media_type="wav", logger=None):
    """
    Prepares the dictionary of parameters for the TTS service.
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
