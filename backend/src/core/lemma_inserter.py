import argparse
import json
import os
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

backend_str = str(BACKEND_DIR)
if backend_str not in sys.path:
    sys.path.insert(0, backend_str)
    
class LemmaInserter:
    def __init__(self):
        with open('vocabularies/nivkh_vocab.json') as file:
            self.vocab = json.load(file)
        with open('vocabularies/morph2gloss.json') as file:
            self.morph2gloss = json.load(file) 
    
    def find_meaning(self, word):
        morphs = []
        if word in self.vocab:
            morphs.extend(self.vocab[word]['rus'])
            if word in self.morph2gloss:
                morphs.extend(self.morph2gloss[word])
        elif word.lower() in self.vocab:
            word = word.lower()
            morphs.extend(self.vocab[word]['rus'])
            if word in self.morph2gloss:
                morphs.extend(self.morph2gloss[word])
        else:
            if word in self.morph2gloss:
                morphs.extend(self.morph2gloss[word])
            else:
                morphs = ['UNK']
        return list(set(morphs))
    
    def gloss_sent_lemmas(self, sent):
        final_sent = []
        for word in sent.split():
            morphemes = word.split('-')
            final_word = []
            for morph in morphemes:
                final_word.append(self.find_meaning(morph))
            final_sent.append(final_word)
        return final_sent
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sent", type=str, required=True, help="String to process")
    args = parser.parse_args()
    text = args.sent
    LEMMA_INSERTER = LemmaInserter()
    glosses = LEMMA_INSERTER.gloss_sent_lemmas(text)
    path_to_write = 'backend/src/core/data/temp'
    os.makedirs(path_to_write, exist_ok=True)
    with open('backend/src/core/data/temp/temp_glossing.json', 'w') as file:
        file.write(json.dumps(glosses, ensure_ascii=False))


if __name__ == '__main__':
    main()

