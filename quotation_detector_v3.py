# -*- coding: utf-8 -*-

import re, string
from collections import Counter
import matplotlib.pyplot as plt
import os
import csv
import glob
import time
from bs4 import BeautifulSoup

start = time.time()

quotes = []
normaltext = []
maybe = []

iterationcounter = 0

word_status = "normal"

quotes_counter = {}
normaltext_counter = {}

errorcount = 0
quotestatus = 0     #toggle parameter to determine whether word is a quote or not

#list of 25 most common stopwords from https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html
#stopwords = ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"]

#https://www.textfixer.com/tutorials/common-english-words.txt
stopwords = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by",
             "can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her",
             "hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most",
             "must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should"
             ,"since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was"
             ,"we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your"]

#excludewords = ["www", "http", "xmlspacepreserve", "wg", "jpg", "png", "ext"]

def remove_reference(wordlist):
    wordlist = [re.sub(r'\[.*?\]', '', word) for word in wordlist]          #remove wiki square bracket with numbers inside
    return wordlist
    

def word_cleaning(wordlist):
    wordlist = [re.compile('[%s]' % re.escape(string.punctuation)).sub('', word) for word in wordlist]          #remove punctuation
    wordlist = [word for word in wordlist if not any(c.isdigit() for c in word)]          #remove words that contain digits
    wordlist = [word for word in wordlist if word not in stopwords]          #remove words that contain digits
    wordlist = [word for word in wordlist if len(word)>=1]                   #remove empty words
    wordlist = [word for word in wordlist if not word[0].isupper()]          #remove words that begin with capital letters
    wordlist = [word.lower() for word in wordlist]                          #set all to lowercase
    return wordlist


def merge_dicts(d1, d2, merge):
    result = dict(d1)
    for k,v in d2.items():
        if k in result:
            result[k] = merge(result[k], v)
        else:
            result[k] = merge(0, v)
    
    for k,v in result.items():
        if k not in d2:
            result[k] = merge(v, 0)
                 
    return result


def subjscore(inputdict, merge):
    score = dict(inputdict)
    for k,v in inputdict.items():
        if sum(v) > 0:
            score[k] = float((v[1]-v[0])/(2*sum(v)) + 0.5)
        else:
            score[k] = 'error'
    return score

def getWords(text):
    return re.compile('\w+').findall(text)

    
#initialize counters for dictionaries
quotes_counter = Counter(dict())
normaltext_counter = Counter(dict())

#path = 'C:\\Users\\WAYNE\\Desktop\\UNI KOBLENZ\\Master Thesis\\Project\\wikipedia_html_dump\\articles_new\\test\\'
path = 'C:\\Users\\WAYNE\\Desktop\\UNI KOBLENZ\\Master Thesis\\Project\\Wikipedia_articles\\wikipedia\\'

for file in glob.glob(os.path.join(path, "*.html")):
    
    inputfile = open(file, 'r', encoding="utf8")
    soup = BeautifulSoup(inputfile, 'html.parser')    
    
    extractedtext = ""
    cleantext = ""
    quotes = []
    normaltext = []
    quotes_counter_file = {}
    normaltext_counter_file = {}
    
    try:  
        #cleantext = soup.find('p').get_text()
        extractedtext = [p.get_text() for p in soup.find_all("p")]
        cleantext = " ".join(str(x) for x in extractedtext)
        #cleantext = soup.find_all('p')
    except AttributeError:
        errorcount = errorcount + 1
        print("Error " + str(errorcount))  

    
    split_text = [word for word in cleantext.split()]
                  
    split_text = remove_reference(split_text)
    
    
    for word in split_text:  

        if len(word) >= 2:      #words of length 1 do not affect word status
            
                if word.startswith('"') and word.endswith('"'):     #single word quote
                    quotes.append(word)
                    word_status = "normal"                          #reset word status to normal after quote
                    
                elif (word.startswith('"') or word[1] == '"'):      #start quotation mark is found
                    if word_status == "normal":
                        word_status = "maybe"                       #word could be the start of a quote
                    elif word_status == "maybe":                    #existing quote exists! this is invalid!
                        normaltext.extend(maybe)
                        maybe = []
                        #word_status = "normal"
                    else:
                        print('Unexpected: <quote> "...' + str(word))
                    
                elif (word.endswith('"') or word[-2] == '"'):        #end quotation mark is found
                    if word_status == "maybe":
                        word_status = "quote"                       #word could be the end of a quote
                    else:
                        word_status = "normal"
                
                else:
                    if word_status == "quote":                      #reset word status to normal after quote
                        word_status = "normal"
                                 
        
            
        if word_status == "normal":
            normaltext.extend(maybe)
            normaltext.append(word)
            maybe = []
        elif word_status == "maybe":
            maybe.append(word)
        elif word_status == "quote":
            quotes.extend(maybe)
            quotes.append(word)
            maybe = []
            word_status = "normal" 
            
 
    #any remaining words in "maybe" at the end of each article will be assumed to be quotations               
    quotes.extend(maybe)
    maybe = []    

    iterationcounter = iterationcounter + 1
    if iterationcounter % 100 == 0:
        print('Iteration ' + str(iterationcounter))           


    #clean words after processing each file
    quotes = word_cleaning(quotes)
    normaltext = word_cleaning(normaltext)
            
    #obtain cleaned word frequencies after processing each file    
    quotes_counter_file = Counter(quotes)
    normaltext_counter_file = Counter(normaltext)
    
    #merge word frequency dict with main counter
    quotes_counter.update(quotes_counter_file)
    normaltext_counter.update(normaltext_counter_file)
    
#merge count values of quote and normal text dicts
merged_output = merge_dicts(normaltext_counter, quotes_counter, lambda x, y:(x,y))
#calculate normalized subjectivity score
#0=fully-objective, 1=fully-subjective
subj_score = subjscore(merged_output, lambda x, y:(x,y))
    


#write to CSV

with open('quotes_counter.csv', 'w', encoding="utf8") as quote_outputfile:
    writer = csv.writer(quote_outputfile)
    writer.writerows(quotes_counter.items())
    quote_outputfile.close()


with open('normaltext_counter.csv', 'w', encoding="utf8") as normal_outputfile:
    writer = csv.writer(normal_outputfile)
    writer.writerows(normaltext_counter.items())
    normal_outputfile.close()


with open('subj_score.csv', 'w', encoding="utf8") as subjscore_outputfile:
    writer = csv.writer(subjscore_outputfile)
    writer.writerows(subj_score.items())
    subjscore_outputfile.close()
    

end = time.time()
#execution time
print(end - start)


