#!/usr/bin/env python3
import argparse

def main(args):
    '''
    Function searches for lines that needs to be changed to be supported by
    PyCTCDecode lib, changes them and writes new KenLM arpa.
    '''
    original = open(args.path_to_ngram, 'r').readlines()
    fixed = open(args.path_to_fixed, 'w')

    for line in original: 
        if 'ngram 1=' in line:
            base_ngram_1_line = line 
            text, value = line.split('=')
            value = str(float(value.replace('\n', ''))+1)
            fixed_ngram_1_line = f"{text}={value}\n"
            fixed.write(fixed_ngram_1_line)
        elif '\t<s>\t' in line: 
            base_token_line = line 
            fixed_token_line = line.replace('\t<s>\t', '\t</s>\t')
            fixed.write(base_token_line)
            fixed.write(fixed_token_line)
        else:
            fixed.write(line)
    fixed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--path_to_ngram", type=str, required=True, help="Path to original KenLM ngram"
    )
    parser.add_argument(
        "--path_to_fixed", type=str, required=True, help="Path to write fixed KenLM ngram"
    )
    args = parser.parse_args()

    main(args)
