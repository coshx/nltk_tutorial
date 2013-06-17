# stress-analyzer.py
# analyzes stress patterns in poems
# uses NLTK and the CMU pronounciation dictionary

import sys
import string
import re

import nltk
from nltk.corpus import cmudict


prondict = cmudict.dict()


def strip_unicode(func):
    """
    decorator for cleanup
    removes non-ASCII characters
    """
    def wrapper(str):
        #this is copied from http://stackoverflow.com/questions/1342000/
        #how-to-replace-non-ascii-characters-in-string
        return "".join(i for i in func(str) if ord(i) < 128)

    return wrapper


def strip_punctuation(func):
    """
    decorator for cleanup
    removes punctuation from a string
    """

    def wrapper(str):
        return func(str).translate(None, string.punctuation)

    return wrapper


def lowercase(func):
    """
    decorator for cleanup
    lowercases string
    """
    
    def wrapper(str):
        return func(str).lower()

    return wrapper


@strip_punctuation
@lowercase
@strip_unicode
def cleanup(str):
    return str


def stress_word(pron):
    """
    get the stress pattern of a word's pronounciation
    this is copied from the NLTK book
    """

    return [char for phone in pron for char in phone if char.isdigit()]


def stress_line(line):
    """get the stress pattern of a line"""
    stress_pattern = []

    #split the line into words (strip away punctuation)
    words = cleanup(line).split()

    #find stress pattern for each word
    #and add it to the stress pattern of the line
    for word in words:
        #find pronounciation of word
        #there might be more than one pronounciation,
        #get the first one for convenience
        if word in prondict:
            pron = prondict[word][0]
            stress_pattern += stress_word(pron)

    return stress_pattern


def stress_poem(poem):
    """return the stress pattern of a poem (i.e., set of lines)"""
    stress_pattern = []
    for line in poem:
        stress_pattern.append(stress_line(line))

    return stress_pattern


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        print stress_poem(f.readlines())

#this is incomplete at the moment
#whenvever a word is not found in the CMU dictionary,
#it is just skipped, which obviously messes with the stress pattern