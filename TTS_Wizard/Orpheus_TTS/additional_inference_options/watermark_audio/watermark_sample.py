from orpheus_tts import OrpheusModel
from watermark import ORPHEUS_WATERMARK, load_watermarker, watermark, verify
import wave
import time
import torch
import torchaudio

def main():
    model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
    watermarker = load_watermarker(device="cuda")
    prompt = "Hello, let's see how well this thing works with a longer generation on cards and watermarks."

    start_time = time.monotonic()
    syn_tokens = model.generate_speech(prompt=prompt, voice="tara")

    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)

        duration = total_frames / wf.getframerate()
        end_time = time.monotonic()
        print(f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio")

if __name__ == '__main__':
    main()

    # Load written audio and move to CUDA
    audio_array, sample_rate = torchaudio.load("output.wav")
    audio_array = audio_array.mean(dim=0).to("cuda")

    # Apply watermark
    watermarker = load_watermarker(device="cuda")
    watermarked_audio, wm_sample_rate = watermark(watermarker, audio_array, sample_rate, ORPHEUS_WATERMARK)
    watermarked_audio = watermarked_audio.cpu()

    # Write watermarked audio
    with wave.open("output_watermarked.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(wm_sample_rate)
        audio_bytes = (watermarked_audio.cpu().numpy() * 32767).astype('int16').tobytes()
        wf.writeframes(audio_bytes)

    # Verify watermark
    is_watermarked = verify(watermarker, watermarked_audio, wm_sample_rate, ORPHEUS_WATERMARK)
    print(f"Watermark verification: {'Success' if is_watermarked else 'Failed'}")