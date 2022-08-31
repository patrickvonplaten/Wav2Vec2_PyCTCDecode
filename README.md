# 🤗 Transformers Wav2Vec2 + PyCTCDecode

## **IMPORTANT** This github repo is not actively maintained. Please try to use: https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM instead**

## UPDATE 2: 

In-detail blog post which should be much better than this repo is available here: https://huggingface.co/blog/wav2vec2-with-ngram

## UPDATE: PyCTCDecode is merged to Transformers!

```diff
import torch
from datasets import load_dataset
from transformers import AutoModelForCTC, AutoProcessor
import torchaudio.functional as F


model_id = "patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm"

sample = next(iter(load_dataset("common_voice", "es", split="test", streaming=True)))
resampled_audio = F.resample(torch.tensor(sample["audio"]["array"]), 48_000, 16_000).numpy()

model = AutoModelForCTC.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

input_values = processor(resampled_audio, return_tensors="pt").input_values

with torch.no_grad():
    logits = model(input_values).logits

-prediction_ids = torch.argmax(logits, dim=-1)
-transcription = processor.batch_decode(prediction_ids)
+transcription = processor.batch_decode(logits.numpy()).text
# => 'bien y qué regalo vas a abrir primero'
```

## Introduction

This repo shows how [🤗 **Transformers**](https://github.com/huggingface/transformers) can be used in combination
with [kensho-technologies's **PyCTCDecode**](https://github.com/kensho-technologies/pyctcdecode) & [**KenLM** ngram](https://github.com/kpu/kenlm) 
as a simple way to boost word error rate (WER).

Included is a file to create an ngram with **KenLM** as well as a simple evaluation script to 
compare the results of using Wav2Vec2 with **PyCTCDecode** + **KenLM** vs. without using any language model.


**Note**: The scripts are written to be used on GPU. If you want to use a CPU instead, 
simply remove all `.to("cuda")` occurances in `eval.py`.

## Installation

In a first step, one should install **KenLM**. For Ubuntu, it should be enough to follow the installation steps 
described [here](https://github.com/kpu/kenlm/blob/master/BUILDING). The installed `kenlm` folder 
should be move into this repo for `./create_ngram.py` to function correctly. Alternatively, one can also 
link the `lmplz` binary file to a `lmplz` bash command to directly run `lmplz` instead of `./kenlm/build/bin/lmplz`.

Next, some Python dependencies should be installed. Assuming PyTorch is installed, it should be sufficient to run
`pip install -r requirements.txt`.

## Run evaluation


### Create ngram

In a first step on should create a ngram. *E.g.* for `polish` the command would be:

```bash
./create_ngram.py --language polish --path_to_ngram polish.arpa
```

After the language model is created, some lines should be converted so it's compatible with 'pyctcdecode'.

Execute the script to run the conversion:

```
./fix_lm.py --path_to_ngram polish.arpa --path_to_fixed polish_fixed.arpa
```

Now the generated 'polish_fixed.arpa' ngram can be correctly used with `pyctcdecode`


### Run eval

Having created the ngram, one can run:

```bash
./eval.py --language polish --path_to_ngram polish.arpa
```

To compare Wav2Vec2 + LM vs. Wav2Vec2 + No LM on polish.


## Results

Without tuning any hyperparameters, the following results were obtained:

```
Comparison of Wav2Vec2 without Language model vs. Wav2Vec2 with `pyctcdecode` + KenLM 5gram.
Fine-tuned Wav2Vec2 models were used and evaluated on MLS datasets.
Take a closer look at `./eval.py` for comparison

==================================================portuguese==================================================
polish - No LM - | WER: 0.3069742867206763 | CER: 0.06054530156286364 | Time: 58.04590034484863
polish - With LM - | WER: 0.2291299753434308 | CER: 0.06211174564528545 | Time: 191.65409898757935

==================================================spanish==================================================
portuguese - No LM - | WER: 0.18208286674132138 | CER: 0.05016682956422096 | Time: 114.61633825302124
portuguese - With LM - | WER: 0.1487761958086706 | CER: 0.04489231909945738 | Time: 429.78511357307434

==================================================polish==================================================
spanish - No LM - | WER: 0.2581272104769545 | CER: 0.0703088156033147 | Time: 147.8634352684021
spanish - With LM - | WER: 0.14927852292116295 | CER: 0.052034208044195916 | Time: 563.0732748508453
```

It can be seen that the word error rate (WER) is significantly improved when using PyCTCDecode + KenLM. 
However, the character error rate (CER) does not improve as much or not at all.
This is expected since using a language model will make sure that words that are predicted are words that exist in the language's vocabulary. 
Wav2Vec2 without a LM produces many words that are more or less correct but contain a couple of spelling errors, thus not contributing to a good WER.
Those words are likely to be "corrected" by Wav2Vec2 + LM leading to an improved WER. However a Wav2Vec2 already has a good character error rate as its 
vocabulary is composed of characters meaning that a "word-based" language model doesn't really help in this case.

Overall WER is probably the more important metric though, so it might make a lot of sense to add a LM to Wav2Vec2. 

In terms of speed, adding a LM significantly reduces speed. However, the script is not at all optimized for speed 
so using multi-processing and batched inference would significantly speed up both Wav2Vec2 without LM and with LM.
