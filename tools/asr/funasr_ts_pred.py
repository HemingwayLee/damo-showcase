import os
from funasr import AutoModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

path_asr  = 'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
path_vad  = 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
path_punc = 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
path_asr  = path_asr  if os.path.exists(path_asr)  else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
path_vad  = path_vad  if os.path.exists(path_vad)  else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_punc = path_punc if os.path.exists(path_punc) else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

# inference_pipline = pipeline(
#    task=Tasks.speech_timestamp,
#    model='iic/speech_timestamp_prediction-v1-16k-offline',
#    model_revision="v2.0.4",
#    output_dir='./tmp')

model = AutoModel(
    model               = "iic/speech_timestamp_prediction-v1-16k-offline",
    model_revision      = "v2.0.4",
#    vad_model           = path_vad,
#    vad_model_revision  = "v2.0.4",
    punc_model          = path_punc,
    punc_model_revision = "v2.0.4"
)
wav_file = f"/workspace/output/slicer/again.wav"
text_file = f"/workspace/tools/asr/asr_output/for_timestamp_matching.txt"
res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
# res = inference_pipline(input=(wav_file, text_file), data_type=("sound", "text"))

print(res)


