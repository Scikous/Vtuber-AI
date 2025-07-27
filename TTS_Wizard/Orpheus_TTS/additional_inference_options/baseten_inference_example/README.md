# Orpheus TTS on Baseten

[Baseten](https://www.baseten.co/) is Canopy Labs' [preferred inference provider](https://www.baseten.co/blog/canopy-labs-selects-baseten-as-preferred-inference-provider-for-orpheus-tts-model) for running Orpheus TTS in production.

## Deployment

To deploy the model, go to [https://www.baseten.co/library/orpheus-tts/](https://www.baseten.co/library/orpheus-tts/) and use the one-click deploy option.

Baseten supports both fp8 (default for performance) and fp16 (full fidelity) versions of Orpheus.

If you want to customize the model serving code, you can instead deploy the prepackaged model from Baseten's [example repository](https://github.com/basetenlabs/truss-examples/tree/main/orpheus-best-performance).



## Inference

The `call_orpheus.py` file contains sample inference code for running the Orpheus TTS model with multiple parallel requests.

Prerequisites:

- Paste the `model_id` from your deployed model into the `call_orpheus.py` script.
- Save your `BASETEN_API_KEY` as an environment variable.

Then, you can call the model with `python call_orpheus.py`.