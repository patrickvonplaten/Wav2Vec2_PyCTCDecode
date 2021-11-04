#!/usr/bin/env python3
from datasets import load_dataset
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--language", default="polish", type=str, required=True, help="Language to run comparison on. Choose one of 'polish', 'portuguese', 'spanish' or add more to this script."
)
parser.add_argument(
    "--path_to_ngram", type=str, required=True, help="Path to kenLM ngram"
)
args = parser.parse_args()

ds = load_dataset("/home/patrick/hugging_face/datasets/datasets/multilingual_librispeech", f"{args.language}", split="train")

with open("text.txt", "w") as f:
    f.write(" ".join(ds["text"]))

os.system(f"./kenlm/build/bin/lmplz -o 5 <text.txt > {args.path_to_ngram}")

## VERY IMPORTANT!!!:
# After the language model is created, one should open the file. one should add a `</s>`
# The file should have a structure which looks more or less as follows:

# \data\
# ngram 1=86586
# ngram 2=546387
# ngram 3=796581
# ngram 4=843999
# ngram 5=850874

# \1-grams:
# -5.7532206      <unk>   0
# 0       <s>     -0.06677356
# -3.4645514      drugi   -0.2088903
# ...

# Now it is very important also add a </s> token to the n-gram
# so that it can be correctly loaded. You can simple copy the line:

# 0       <s>     -0.06677356

# and change <s> to </s>. When doing this you should also inclease `ngram` by 1.
# The new ngram should look as follows:

# \data\
# ngram 1=86587
# ngram 2=546387
# ngram 3=796581
# ngram 4=843999
# ngram 5=850874

# \1-grams:
# -5.7532206      <unk>   0
# 0       <s>     -0.06677356
# 0       </s>     -0.06677356
# -3.4645514      drugi   -0.2088903
# ...

# Now the ngram can be correctly used with `pyctcdecode`
