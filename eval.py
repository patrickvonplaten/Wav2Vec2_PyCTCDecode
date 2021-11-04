#!/usr/bin/env python3
import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor
import time
import argparse

from datasets import load_dataset, load_metric
from pyctcdecode import build_ctcdecoder


LANG_TO_ID_LOOK_UP = {
    "polish": "pl",
    "portuguese": "pt",
    "spanish": "es",
}


def main(args):
    language = args.language
    lang_id = LANG_TO_ID_LOOK_UP[language]

    if lang_id == "pt":
        model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-portuguese").to("cuda")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-portuguese")
    else:
        model = AutoModelForCTC.from_pretrained(f"facebook/wav2vec2-base-10k-voxpopuli-ft-{lang_id}").to("cuda")
        processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-base-10k-voxpopuli-ft-{lang_id}")

    wer = load_metric("wer")
    cer = load_metric("cer")

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    decoder = build_ctcdecoder(
        list(sorted_dict.keys()),
        args.path_to_ngram,
    )

    # load trained kenlm model
    ds = load_dataset("multilingual_librispeech", f"{language}", split="test")

#    Uncomment for dummy run
#    ds = ds.select(range(20))

    def map_to_wer_no_lm(batch):
        input_values = processor(batch["audio"]["array"], return_tensors="pt", sampling_rate=16_000).input_values.to("cuda")

        with torch.no_grad():
            pred_ids = torch.argmax(model(input_values).logits, -1)

        pred_str = processor.batch_decode(pred_ids)

        batch["pred_str"] = pred_str[0]
        batch["ref_str"] = batch["text"]
        return batch

    def map_to_wer_with_lm(batch):
        input_values = processor(batch["audio"]["array"], return_tensors="pt", sampling_rate=16_000).input_values.to("cuda")

        with torch.no_grad():
            logits = model(input_values).logits.cpu().numpy()[0]

        batch["pred_str"] = decoder.decode(logits)
        batch["ref_str"] = batch["text"]

        return batch

    start_time_1 = time.time()
    result_no_lm = ds.map(map_to_wer_no_lm, remove_columns=ds.column_names)

    wer_result_no_lm = wer.compute(predictions=result_no_lm["pred_str"], references=result_no_lm["ref_str"])
    cer_result_no_lm = cer.compute(predictions=result_no_lm["pred_str"], references=result_no_lm["ref_str"])

    start_time_2 = time.time()
    result_with_lm = ds.map(map_to_wer_with_lm, remove_columns=ds.column_names)

    wer_result_with_lm = wer.compute(predictions=result_with_lm["pred_str"], references=result_with_lm["ref_str"])
    cer_result_with_lm = cer.compute(predictions=result_with_lm["pred_str"], references=result_with_lm["ref_str"])

    end_time = time.time()

    print(50 * "=" + language + 50 * "=")
    print(f"{language} - No LM - | WER: {wer_result_no_lm} | CER: {cer_result_no_lm} | Time: {start_time_2 - start_time_1}")
    print(f"{language} - With LM - | WER: {wer_result_with_lm} | CER: {cer_result_with_lm} | Time: {end_time - start_time_2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--language", default="polish", type=str, required=True, help="Language to run comparison on. Choose one of 'polish', 'portuguese', 'spanish' or add more to this script."
    )
    parser.add_argument(
        "--path_to_ngram", type=str, required=True, help="Path to kenLM ngram"
    )
    args = parser.parse_args()

    main(args)
