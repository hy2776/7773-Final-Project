
import re
import numpy as np
import pandas as pd
import os

def cleanStatement(statement):
    
    def getReplacementList (list_name):
        allWords = [line.rstrip('\n') for line in  open(list_name, 'r') ]
        oldWords = [allWords[i] for i in range(len(allWords)) if i % 2 == 0]
        newWords = [allWords[i] for i in range(len(allWords)) if i % 2 == 1]
        return [oldWords, newWords]
    
    wordlistdir = '../data/Wordlist'
    replacements = getReplacementList(os.path.join(wordlistdir,"wordlist.txt"))
    stoplist = [line.rstrip('\n') for line in \
                      open(os.path.join(wordlistdir,"stoplist_mcdonald_comb.txt"), 'r') ]
    # Read in the statement and convert it to lower case
    original  = statement.lower()

    clean = original
    # Remove punctuation and newlines first, to keep space between words
    for todelete in ['-', '|', '/', ',', '.', ':', ';']:
        clean = clean.replace(todelete, ' ')

    # Regular expression to find a paragraph starting with 'For immediate release' or 'For release at'
    deleteBefore_pattern = r"([Ff]or\s([Ii]mmediate\s[Rr]elease|[Rr]elease\sat).*?)(\n|\Z)"
    deleteBefore_match = re.search(deleteBefore_pattern, clean, re.DOTALL)

    # If a match is found, delete the matched paragraph and everything before it
    if deleteBefore_match:
        delete_before = deleteBefore_match.end()
        clean = clean[delete_before:]

    deleteAfter_match = re.search("[Lr]ast\s[Uu]pdate", clean)
    if deleteAfter_match:
        delete_after = deleteAfter_match.start()
        clean = clean[:delete_after]

    # Looking for the end of the text
    intaking   = re.search("in\staking\sthe\sdiscount\srate\saction",\
                           clean)
    votingfor  = re.search("voting\sfor\sthe", clean)
    if intaking == None and not votingfor == None:
        deleteAfter = votingfor.start()
    elif votingfor == None and not intaking == None:
        deleteAfter = intaking.start()
    elif votingfor == None and intaking == None:
        deleteAfter = len(clean)
    else:
        deleteAfter = min(votingfor.start(), intaking.start())
    clean = clean[:deleteAfter]

    for todelete in ['\r\n', '\n']:
        clean = clean.replace(todelete, ' ')

    # Replace replacement words (concatenations)
    for word in range(len(replacements[0])):
        clean = clean.replace(replacements[0][word], replacements[1][word])

    # Remove stop words
    for word in stoplist:
        clean = clean.replace(' '+word.lower() + ' ', ' ')
    
    # Keep only the characters that you want to keep
    charsTokeep = '[^A-Za-z ]+'
    clean = re.sub(charsTokeep, '', clean)
    # clean = re.sub(r'\d+', '', clean)
    clean = clean.replace('section',' ')
    clean = clean.replace('  ', ' ')
    clean = clean.replace(' u s ', ' unitedstates ')

    deleteAfter_match = re.search("home\spress\sreleases\saccessibility\scontact\supdate", clean)
    if deleteAfter_match:
        delete_after = deleteAfter_match.start()
        clean = clean[:delete_after]
    
    deleteAfter_match = re.search("home\snews\sevents\saccessibility\supdate", clean)
    if deleteAfter_match:
        delete_after = deleteAfter_match.start()
        clean = clean[:delete_after]

    return clean

def get_sentiment_from_pretrained_model(statement):
    from transformers import BertTokenizer, BertForSequenceClassification, pipeline

    # Load the model and tokenizer from local paths
    model_path = '../data/models/pretrained_sentiment_model/'
    tokenizer_path = '../data/models/tokenizer/'
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Initialize the pipeline
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)

    # apply the sentiment analysis model to a statement
    result = nlp(statement)
    return result[0]['label'],result[0]['score']


def get_sentiment_score_LM(statement):
    import pickle
    with open('../data/LoughranMcDonald/sentiment_dictionaries.pkl', 'rb') as file:
        sentiment_dictionaries = pickle.load(file)

    lmdict = {'Negative': [x.lower() for x in sentiment_dictionaries['negative'].keys()],
            'Positive': [x.lower() for x in sentiment_dictionaries['positive'].keys()]}
    negate = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't",
            "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt",
            "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't",
            "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "wasnt",
            "werent", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "without", "wont", "wouldnt", "won't",
            "wouldn't", "rarely", "seldom", "despite", "no", "nobody"]
    
    def negated(word):
        """
        Determine if preceding word is a negation word
        """
        if word.lower() in negate:
            return True
        else:
            return False
    
    def tone_count_with_negation_check(dict, article):
        """
        Count positive and negative words with negation check. Account for simple negation only for positive words.
        Simple negation is taken to be observations of one of negate words occurring within three words
        preceding a positive words.
        """
        pos_count = 0
        neg_count = 0
    
        pos_words = []
        neg_words = []
    
        input_words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', article.lower())
    
        word_count = len(input_words)
    
        for i in range(0, word_count):
            if input_words[i] in dict['Negative']:
                neg_count += 1
                neg_words.append(input_words[i])
            if input_words[i] in dict['Positive']:
                if i >= 3:
                    if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                        neg_count += 1
                        neg_words.append(input_words[i] + ' (with negation)')
                    else:
                        pos_count += 1
                        pos_words.append(input_words[i])
                elif i == 2:
                    if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                        neg_count += 1
                        neg_words.append(input_words[i] + ' (with negation)')
                    else:
                        pos_count += 1
                        pos_words.append(input_words[i])
                elif i == 1:
                    if negated(input_words[i - 1]):
                        neg_count += 1
                        neg_words.append(input_words[i] + ' (with negation)')
                    else:
                        pos_count += 1
                        pos_words.append(input_words[i])
                elif i == 0:
                    pos_count += 1
                    pos_words.append(input_words[i])
    
        results = [word_count, pos_count, neg_count, pos_words, neg_words]
    
        return results
    
    res = tone_count_with_negation_check(lmdict, statement)

    # Sentiment Score normalized by the number of words
    # (NPositiveWords - NNegtiveWords) / wordcount * 100
    senti_score = (res[1] - res[2]) / res[0] * 100
    return senti_score

