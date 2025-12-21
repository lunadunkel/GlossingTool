import json

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
            final_sent.append(final_sent)
            print(final_word)
        return final_sent