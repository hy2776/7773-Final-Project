# Description: This file converts the raw FOMC Statements into cleaner
#              versions of themselves.
#
#              This file leverages an open source version for the cleaning.
#              The content has been modified from it's originally published version and
#              adapted for python3. The author of that original version is
#              Miguel Acosta  www.acostamiguel.com
#
# Input:       The raw FOMC statements, downloaded from federalreserve.gov
#              by the python script pullStatements.py. These are
#              located in the directory statements/statements.raw
#
# Output:      Two sets of cleaned FOMC statements:
#                (i) A set, located in the directory statements/statements.clean
#                    of FOMC statements that have had header, footer, and voting
#                    information removed. These files are currently being used
#                    in this project.
#                (ii) A set, located in statements/statements.clean.np
#                    of FOMC statements that have had header, footer, and voting
#                    information removed. They have also been stemmed, words
#                    have been concatenated, and numbers/stopwords
#                    have been removed. These files are not currently being used
#                    in this project.
#
#--------------------------------- IMPORTS -----------------------------------#
import os, csv, re
from os import listdir
from os.path import isfile, join
# from nltk.stem.lancaster import LancasterStemmer
# from textmining_withnumbers import TermDocumentMatrix as TDM

#-------------------------DEFINE GLOBAL VARIABLES-----------------------------#
# Directory where the stop words and n-grams to concatenate are
datadir      = '../data/FOMC'
wordlistdir  = '../data/wordlist'
# Where the raw statements are
statementdir = os.path.join(datadir,'statement')
# Where the clean statements will go (with and without preprocessing)
cleanDir     = os.path.join(datadir,'statements.clean')
cleanDirNP   = os.path.join(datadir,'statements.clean.np')

#-----------------------------------------------------------------------------#
# getReplacementList: Returns two lists, a list of N n-grams (phrase with n
#   words) and a list with N "words" to replace the n-grams. The function reads
#   a file, list_name, where every odd entry is an n-gram, and every even entry
#   is a replacement.
#-----------------------------------------------------------------------------#
def getReplacementList (list_name):
    allWords = [line.rstrip('\n') for line in  open(list_name, 'r') ]
    oldWords = [allWords[i] for i in range(len(allWords)) if i % 2 == 0]
    newWords = [allWords[i] for i in range(len(allWords)) if i % 2 == 1]
    return [oldWords, newWords]

#-----------------------------------------------------------------------------#
# cleanStatement: This function is the meat of this code--it performs all of the
#   cleaning/preprocessing described in the header of this document. It's
#   inputs are:
#     (1) statement   : a string with the filename of a single FOMC statement
#     (2) locationold : Directory where raw statements are located (string)
#     (3) replacements: Output from getReplacementList
#     (4) locationnew : Directory where clean statements go (string)
#     (5) stoplist    : A list of words to remove (list of strings)
#     (6) charsToKeep : A regular expression of the character types to keep
#-----------------------------------------------------------------------------#
def cleanStatement (statement, locationold, replacements, locationnew, \
                    stoplist):
    # Read in the statement and convert it to lower case
    original  = open(os.path.join(locationold,statement),'r').read().lower()

    clean = original
    # Remove punctuation and newlines first, to keep space between words
    for todelete in ['-', '|', '/', ',', '.', ':', ';']:
        clean = clean.replace(todelete, ' ')



    # Remove anything before (and including) 'for immediate release'

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
        
    

    # Write cleaned file
    new = open(os.path.join(locationnew,statement), 'w')
    new.write(clean)
    new.close

#-----------------------------------------------------------------------------#
# The Main function generates the stop list, and word replacement lists, then
#   loops through every file in the statements/statements.raw directory and
#   performs two types of cleaning: one that is less extensive (saved in
#   statements/statements.clean) and one that includes more preprocessing steps
#   (saved in statements/statements.clean.np). Finally, it creates the
#   term-document matrix for each type of cleaning. 'NP' denotes 'no preprocessing.
#
#   Only the files in the statements/statements.clean folder are currently
#   being used in this project, these are the files with less preprocessing
#   at this step. More processing happens in the Jupyter Notebook code which
#   is the file called Data.ipynb.
#-----------------------------------------------------------------------------#

def main():
    stoplist       = [line.rstrip('\n') for line in \
                      open(os.path.join(wordlistdir,"stoplist_mcdonald_comb.txt")
                           , 'r') ]
    stoplistNP     = [line.rstrip('\n') for line in \
                      open(os.path.join(wordlistdir,"emptystop.txt"), 'r') ]

    replacements   = getReplacementList(os.path.join(wordlistdir,"wordlist.txt"))
    replacementsNP = getReplacementList(os.path.join(wordlistdir,"wordlist.np.txt"))

    statementList  = [ f for f in listdir(statementdir) \
                       if isfile(join(statementdir,f)) ]

    for statement in statementList:
        # First, the case with heavier preprocessing (keep only letters)
        cleanStatement(statement, statementdir, replacements, \
                       cleanDir, stoplist)
        # Second, the no-preprocessing case (keep letters and numbers)
        cleanStatement(statement, statementdir, replacementsNP, \
                       cleanDirNP, stoplistNP)

if __name__ == "__main__":
    main()
