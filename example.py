# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/api.html#modules
# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/results.html#inference-with-multi-task-models

from nemo.collections.asr.models import EncDecMultiTaskModel
import wget
wget.download("https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav")
# load model
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

# update dcode params
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)

canary_model.transcribe(
        audio=['2086-149220-0033.wav'],
        batch_size=4,  # batch size to run the inference with
        task="asr",  # use "ast" for speech-to-text translation
        source_lang="en",  # language of the audio input, set `source_lang`==`target_lang` for ASR
        target_lang="en",  # language of the text output
        pnc='true',  # whether to have PnC output, choices=[True, False]
)


# Potentially also tts
# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/api.html