# https://medium.com/@oduguwadamilola40/byte-pair-encoding-the-tokenization-algorithm-powering-large-language-models-5055fbdc0153
from collections import defaultdict

with open("LLM data.txt", "r", encoding="utf-8") as f:
    corpus = [i.strip() for i in f.readlines()]
    corpus.append(" ")

word_freq = defaultdict(int)
for sent in corpus:
    for word in sent.split():
        if word in word_freq.keys():
            word_freq[word] += 1

        else:
            word_freq[word] = 1

base_vocab = sorted(list(set(list(" ".join(corpus)))))
splits = {word: [char for char in word] for word in word_freq.keys()}

def find_pair_freq():
    pair_freq = defaultdict(int)
    for word, freq in word_freq.items():
        split = splits[word]
        if len(split) < 1:
            continue

        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_freq[pair] += freq

    return pair_freq

def merge(a, b, splits):
    for word in word_freq.keys():
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a + b] + split[i+2:]

            else:
                i += 1

        splits[word] = split
    return splits

vocab_size = 250
while len(base_vocab) < vocab_size:
    pair_freq = find_pair_freq()
    best_pair = ""
    max_freq = None

    for pair, freq in pair_freq.items():
        if max_freq == None or max_freq < freq:
            max_freq = freq
            best_pair = pair

        splits = merge(*best_pair, splits=splits)
        base_vocab.append(best_pair[0] + best_pair[1])

print(base_vocab)
