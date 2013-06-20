# classifier_chunker.py

import nltk
from nltk.chunk.util import conlltags2tree


def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == "DT":
            tags = set()
        else:
            tags.add(pos)

    return "+".join(sorted(tags))


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]

    if i == 0:
        prevword, prevpos = "<START>", "START"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]

    return {
        "pos": pos,
        "word": word,
        "prevpos": prevpos,
        "nextpos": nextpos,
        "prevpos+pos": "{0}+{1}".format(prevpos, pos),
        "pos+nextpos": "{0}+{1}".format(pos, nextpos),
        "tags_since_dt": tags_since_dt(sentence, i)
    }


class ConsecutiveNPChunkTagger(nltk.TaggerI):
    
    def __init__(self, train_sents):
        train_set = []

        #extract feature set from each sentence
        for chunked_sent in train_sents:
            #strip list of word-tags from word-tag and chunk tuple
            unchunked_sent = nltk.tag.untag(chunked_sent)
            history = []

            #build features for each word-tag tuple
            for i, (wordtag, chunk) in enumerate(chunked_sent):
                feature_set = npchunk_features(unchunked_sent, i, history)
                #add word feature and tag tuple to training set
                train_set.append( (feature_set, chunk) )
                #add tag to history (needed for building word featureset)
                history.append(chunk)

        #feed the training set to a maximum entropy classifier
        self.classifier = nltk.MaxentClassifier.train(train_set, algorithm="GIS")

    def tag(self, sentence):
        history = []
        #extract featureset from each word-tag
        for i, wordtag in enumerate(sentence):
            feature_set = npchunk_features(sentence, i, history)
            chunk = self.classifier.classify(feature_set)
            history.append(chunk)

        #return list of word-tag tuple
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):

    def __init__(self, train_sents):
        #convert to word-tag and chunk tuple list
        chunked_sents = [ [((word, tag), chunk)
            for (word, tag, chunk) in nltk.chunk.tree2conlltags(sent)]
            for sent in train_sents]

        #train tagger
        self.tagger = ConsecutiveNPChunkTagger(chunked_sents)

    def parse(self, sentence):
        #classify chunks for list of word-tags
        chunked_sents = self.tagger.tag(sentence)

        #convert to tree
        return conlltags2tree([(word, tag, chunk)
            for ((word, tag), chunk) in chunked_sents])
