import os
from funasr import AutoModel

path_vad  = 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
path_vad  = path_vad  if os.path.exists(path_vad)  else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


model = AutoModel(
    model=path_vad,
    model_revision="v2.0.4"
)
wav_file = f"/workspace/output/slicer/bean.wav"
res = model.generate(input=wav_file)
print(res)

