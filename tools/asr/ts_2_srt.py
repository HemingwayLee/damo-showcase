import os
import json
from datetime import timedelta


# from funasr import AutoModel
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks

# path_asr  = 'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
# path_punc = 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
# path_asr  = path_asr  if os.path.exists(path_asr)  else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# path_punc = path_punc if os.path.exists(path_punc) else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

# model = AutoModel(
#     model               = "iic/speech_timestamp_prediction-v1-16k-offline",
#     model_revision      = "v2.0.4",
#     punc_model          = path_punc,
#     punc_model_revision = "v2.0.4"
# )
# wav_file = f"/workspace/output/slicer/again.wav"
# text_file = f"/workspace/tools/asr/asr_output/for_timestamp_matching.txt"
# res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))

with open('./data/57-3.txt', encoding="utf-8") as fp:
    contents = fp.read()
    # print(contents)

with open('./data/57-3.json', encoding='utf-8') as fp:
    data = json.load(fp)
    # print(data)

pred_txt = data["text"].split()
tss = data['timestamp']

# No idea why the first element will be in the key
first = data["key"]
# pred_txt.insert(0, first) 

print(f"len: {len(contents.replace('？', '').replace('、', '').replace('。', '').replace('，', '').split())}")
print(f"len: {len(tss)}")
print(f"len: {len(pred_txt)}")

res = []
sentence = ""
j = 0
start_ts = 0
end_ts = 0
for i, char in enumerate(contents.split()):
    # print(f"{i}: {char}")
    if i == 0:
        sentence = sentence + first
        continue

    if char == "。" or char == "，" or char == "、" or char == "？" or char == "!":
        res.append({
            "start": str(timedelta(milliseconds=start_ts)),
            "end": str(timedelta(milliseconds=end_ts)),
            "sentence": sentence
        })

        start_ts = -1
        end_ts = -1
        sentence = ""

        continue

    sentence = sentence + char

    if pred_txt[j] != char:
        print("not working!!")

    if start_ts == -1:
        start_ts = tss[j][0]

    # print(tss[j])
    end_ts = tss[j][1]

    j = j + 1

print("final res:")
print(res)

with open('final_res.srt', "w") as fp:
    for idx, ele in enumerate(res):
        fp.write(f'{idx}\n')
        fp.write(f'{ele["start"]} --> {ele["end"]}\n')
        fp.write(f'{ele["sentence"]}\n')
        fp.write('\n')
