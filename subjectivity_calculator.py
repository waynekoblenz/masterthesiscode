# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:00:57 2017

@author: Pang Wayne
"""
import numpy as np
import pandas as pd
import re, string
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import scipy.stats as stats
from scipy.stats import norm
from statistics import mean, stdev
from math import sqrt
from string import digits
import itertools
import codecs
from nltk.corpus import words


subjscore = 0
reviewlength = 0
unknownwords = 0
stopwordcount = 0


#common English stopwords
#https://www.textfixer.com/tutorials/common-english-words.txt
stopwords = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by",
             "can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her",
             "hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most",
             "must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should"
             ,"since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was"
             ,"we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your"]

             
def word_cleaning(word):
    word = re.sub('&quot;', '', word)
    word = re.sub('&gt', '', word)
    word = re.sub('&lt', '', word)  
    word = re.sub('{{cite', '', word)
    word = word.lower()
    word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)     #exclude all punctuations
    return word

def compareresults(df, median):
  if (df['SUBJSCORE'] < median and df['SUBJTYPE'] == 'weaksubj') |  df['SUBJSCORE'] > median and df['SUBJTYPE'] == 'strongsubj':
    return 'True'
  else:
    return 'False'
    
def hasNumbers(inputString):
    return any(char.isdigit() for char in str(inputString))    

def removedigits(cell_content):
    if hasNumbers(cell_content):
        cell_content.str.replace('\d+', '')
    return cell_content



print('\nSUBJECTIVITY SCORE\n')
data = pd.read_csv('subj_score.csv', names = ["WORD","SUBJSCORE"])
if 'NaN' in data.index:
    data.drop(['NaN'])
print(data.head())


print('\nNORMAL TEXT COUNTER\n')
normaltext_counter =  pd.read_csv('normaltext_counter.csv', names = ["WORD","FREQUENCY_NORMAL"])
if 'NaN' in normaltext_counter:
    normaltext_counter.drop(['NaN'])
normaltext_counter_sorted = normaltext_counter.sort_values(by='FREQUENCY_NORMAL', ascending=False)
print(normaltext_counter_sorted.head(30))
#statistical measure of word frequency
print('Mean:')
print(normaltext_counter["FREQUENCY_NORMAL"].mean())
print('Median:')
print(normaltext_counter["FREQUENCY_NORMAL"].median())



print('\nQUOTED TEXT COUNTER\n')
quotes_counter =  pd.read_csv('quotes_counter.csv', names = ["WORD","FREQUENCY_QUOTE"])
if 'NaN' in quotes_counter.index:
    #quotes_counter.drop(['NaN'])
    quotes_counter.drop(quotes_counter.index['NaN'])
quotes_counter_sorted = quotes_counter.sort_values(by='FREQUENCY_QUOTE', ascending=False)
print(quotes_counter_sorted.head(30))
#statistical measure of word frequency
print('Mean:')
print(quotes_counter["FREQUENCY_QUOTE"].mean())
print('Median:')
print(quotes_counter["FREQUENCY_QUOTE"].median())


#merging frequencies of quotes and normal words into subjscore dataframe
df_merged = data.merge(normaltext_counter, how='outer')
#print(df_merged.head())
df_merged_all = df_merged.merge(quotes_counter, how='outer')
#print(df_merged_all.head())

df_merged_all = df_merged_all[df_merged_all['WORD'].isin(words.words())]


#filter out words that have too low total frequency [MINIMUM THRESHOLD = 10]
data_frequency_threshold = df_merged_all[(df_merged_all['FREQUENCY_QUOTE'] >= 10) | (df_merged_all['FREQUENCY_NORMAL'] >= 10)]


#show top subjective words
data_frequency_threshold_sorted = data_frequency_threshold.sort_values(by = ['SUBJSCORE', 'FREQUENCY_QUOTE'], ascending=[False, False])
print('\nTop subjective words sorted by frequency:\n')
print(data_frequency_threshold_sorted.head(30))

#show top objective words
data_frequency_threshold_sorted = data_frequency_threshold.sort_values(by = ['SUBJSCORE', 'FREQUENCY_NORMAL'], ascending=[True, False])
print('\nTop objective words sorted by frequency:\n')
print(data_frequency_threshold_sorted.head(30))

print('Total frequency of quote words:')
print(data_frequency_threshold['FREQUENCY_QUOTE'].sum())
print('Total frequency of normal words:')
print(data_frequency_threshold['FREQUENCY_NORMAL'].sum())

#distribution of subjectivity scores (no threshold applied)
plt.title('Histogram of Quote Scores (no minimum frequency threshold)')
n, bins, patches = plt.hist(data['SUBJSCORE'], 20, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.0)
plt.gca().set_yscale("log")
plt.grid(True)
plt.show()

#distribution of subjectivity scores for words with frequency >= threshold
plt.title('Histogram of Quote Scores (frequency >= threshold)')
n, bins, patches = plt.hist(data_frequency_threshold['SUBJSCORE'], 20, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.0)
plt.grid(linestyle='dashed', linewidth=0.5)
plt.gca().set_yscale("log")
plt.grid(True)
plt.show()

#calculate skewness
print('Skew test of quote scores:')
print(stats.skewtest(data_frequency_threshold["SUBJSCORE"]))

#probablity plot
fig = plt.figure()
stats.probplot(data_frequency_threshold["SUBJSCORE"], plot=plt, rvalue=True)
plt.show()

#statistical measure of word frequency
print('Mean:')
print(data_frequency_threshold["SUBJSCORE"].mean())
print('Median:')
print(data_frequency_threshold["SUBJSCORE"].median())
print('Standard Deviation:')
print(data_frequency_threshold["SUBJSCORE"].std())


print('\n#################################### MPQA ####################################\n')

#read MPQA lexicon
mpqa_input = pd.read_csv('mpqa.txt', sep=" ", header=None, error_bad_lines=False, names = ["SUBJTYPE","LEN","WORD","TYPE","STEM","PRIOR"])
#print(mpqa_input.head())

#extract only words and subjectivity classification from lexicon
mpqa_lexicon = mpqa_input[['WORD', 'SUBJTYPE']].copy()
mpqa_lexicon['WORD'] = mpqa_lexicon['WORD'].apply(lambda x : x[6:])
mpqa_lexicon['SUBJTYPE'] = mpqa_lexicon['SUBJTYPE'].apply(lambda x : x[5:])

mpqa_duplicates = mpqa_lexicon[mpqa_lexicon.duplicated(keep=False)]

mpqa_lexicon = mpqa_lexicon.drop_duplicates(subset=['WORD', 'SUBJTYPE'], keep=False)
#print(mpqa_lexicon.head(30))

#merge word subjectiviy scores and frequencies with MPQA lexicon
df_merged_all_with_mpqa = mpqa_lexicon.merge(data_frequency_threshold, how='outer')
df_merged_all_with_mpqa = df_merged_all_with_mpqa[pd.notnull(df_merged_all_with_mpqa['SUBJSCORE'])]
df_merged_all_with_mpqa = df_merged_all_with_mpqa[pd.notnull(df_merged_all_with_mpqa['SUBJTYPE'])]
#df_merged_all_with_mpqa = df_merged_all_with_mpqa[(df_merged_all_with_mpqa['FREQUENCY_QUOTE'] >= 5) | (df_merged_all_with_mpqa['FREQUENCY_NORMAL'] >= 5)]
df_merged_all_with_mpqa = df_merged_all_with_mpqa.sort_values(by = ['SUBJSCORE', 'FREQUENCY_QUOTE'], ascending=[False, False])
#df_merged_all_with_mpqa.groupby('WORD').mean().reset_index()

print('Words that are common between MPQA and Wikipedia text corpus:')
print(df_merged_all_with_mpqa.head())

print('\n')

#calculate median
mpqa_wiki_median = df_merged_all_with_mpqa["SUBJSCORE"].median()
print('Median (MPQA):')
print(mpqa_wiki_median)

#calculate mean
mpqa_wiki_mean = df_merged_all_with_mpqa["SUBJSCORE"].mean()
print('Mean (MPQA):')
print(mpqa_wiki_mean)

#calculate standard deviation
mpqa_wiki_std = df_merged_all_with_mpqa["SUBJSCORE"].std()
print('Standard deviation (MPQA):')
print(mpqa_wiki_std)

print('\n')

#get words which are classified as "strongsubj" in MPQA
print('STRONGSUBJ Words that are common between MPQA and Wikipedia text corpus:')
strong_merged_all_with_mpqa = df_merged_all_with_mpqa.loc[df_merged_all_with_mpqa['SUBJTYPE'] == "strongsubj"]
print(strong_merged_all_with_mpqa.head())

print('\n')

print('Statistical measures of STRONGSUBJ')
#calculate median
strong_mpqa_wiki_median = strong_merged_all_with_mpqa["SUBJSCORE"].median()
print('Median:')
print(strong_mpqa_wiki_median)

#calculate mean
strong_mpqa_wiki_mean = strong_merged_all_with_mpqa["SUBJSCORE"].mean()
print('Mean:')
print(strong_mpqa_wiki_mean)

#calculate standard deviation
strong_mpqa_wiki_std = strong_merged_all_with_mpqa["SUBJSCORE"].std()
print('Standard Deviation:')
print(strong_mpqa_wiki_std)

#display boxplot
plt.figure(1)
plt.title("Boxplot of SUBJSCORE for words classified as SUBJSTRONG in MPQA")
plt.ylim([0,1])
strong_merged_all_with_mpqa.boxplot(column=['SUBJSCORE'])
#strong_merged_all_with_mpqa.boxplot(column=['SUBJSCORE'], return_type='axes')

print('\n')

#get words which are classified as "weaksubj" in MPQA
print('WEAKSUBJ Words that are common between MPQA and Wikipedia text corpus:')
weak_merged_all_with_mpqa = df_merged_all_with_mpqa.loc[df_merged_all_with_mpqa['SUBJTYPE'] == "weaksubj"]
print(weak_merged_all_with_mpqa.head())

print('\n')

#calculate median
print('Statistical measures of WEAKSUBJ')
weak_mpqa_wiki_median = weak_merged_all_with_mpqa["SUBJSCORE"].median()
print('Median:')
print(weak_mpqa_wiki_median)

#calculate mean
weak_mpqa_wiki_mean = weak_merged_all_with_mpqa["SUBJSCORE"].mean()
print('Mean:')
print(weak_mpqa_wiki_mean)

#calculate standard deviation
weak_mpqa_wiki_std = weak_merged_all_with_mpqa["SUBJSCORE"].std()
print('Standard Deviation:')
print(weak_mpqa_wiki_std)

#display boxplot
plt.figure(2)
plt.title("Boxplot of SUBJSCORE for words classified as SUBJWEAK IN MPQA")
plt.ylim([0,1])
weak_merged_all_with_mpqa.boxplot(column=['SUBJSCORE'])
#strong_merged_all_with_mpqa.boxplot(column=['SUBJSCORE'], return_type='axes')

print('\n')

#histogram for words in MPQA
plt.figure(3)
plt.title('Histogram of Quote Scores in MQPA lexicon')
n, bins, patches = plt.hist(df_merged_all_with_mpqa['SUBJSCORE'], 20, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.0)
plt.grid(linestyle='dashed', linewidth=0.5)
plt.grid(True)
plt.show()

print('\n')

#inspecting the distribution
#calculate chi square
mpqa_wiki_chisquare = stats.normaltest(df_merged_all_with_mpqa["SUBJSCORE"])
print('Test for normality (MPQA):')
print(mpqa_wiki_chisquare)

#calculate kurtosis (flatness/peakiness)
mpqa_wiki_kurtosis = stats.kurtosistest(df_merged_all_with_mpqa["SUBJSCORE"])
print('Kurtosis test (MPQA):')
print(mpqa_wiki_kurtosis)

#calculate skewness (slant left/right)
mpqa_wiki_skew = stats.skewtest(df_merged_all_with_mpqa["SUBJSCORE"])
print('Skew test (MPQA):')
print(mpqa_wiki_skew)

#probablity plot
fig = plt.figure()
stats.probplot(df_merged_all_with_mpqa["SUBJSCORE"], plot=plt, rvalue=True)
plt.show()


#histogram for SUBJSTRONG words in MPQA
plt.figure(4)
plt.title('Histogram of Quote Scores for words classified as "strongsubj" in MQPA Lexicon')
plt.xlim([0,1])
plt.ylim([0,12])
n, bins, patches = plt.hist(strong_merged_all_with_mpqa['SUBJSCORE'], 20, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.0)
plt.grid(linestyle='dashed', linewidth=0.5)
plt.grid(True)
plt.show()

#histogram for WEAKSUBJ words in MPQA
plt.figure(5)
plt.title('Histogram of Quote Scores for words classified as "weaksubj" in MQPA Lexicon')
plt.xlim([0,1])
plt.ylim([0,12])
n, bins, patches = plt.hist(weak_merged_all_with_mpqa['SUBJSCORE'], 20, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.0)
plt.grid(linestyle='dashed', linewidth=0.5)
plt.grid(True)
plt.show()



print('\n################################## non-MPQA ##################################\n')

#get words in my lexicon which are NOT in MPQA lexicon
cols = ['WORD','SUBJSCORE']
#get copies where the indices are the columns of interest
df2 = data_frequency_threshold.set_index(cols)
other2 = df_merged_all_with_mpqa.set_index(cols)
#look for index overlap
df_merged_all_notin_mpqa = data_frequency_threshold[~df2.index.isin(other2.index)]
print(df_merged_all_notin_mpqa.head())

print('\n')

#calculate mean
notin_mpqa_wiki_mean = df_merged_all_notin_mpqa["SUBJSCORE"].mean()
print('Mean (Words not in MPQA):')
print(notin_mpqa_wiki_mean)

#calculate median
notin_mpqa_wiki_median = df_merged_all_notin_mpqa["SUBJSCORE"].median()
print('Median (Words not in MPQA):')
print(notin_mpqa_wiki_median)

#calculate standard deviation
notin_mpqa_wiki_std = df_merged_all_notin_mpqa["SUBJSCORE"].std()
print('Standard deviation (Words not in MPQA):')
print(notin_mpqa_wiki_std)


#histogram for words NOT in MPQA lexicon
plt.title('Histogram of Quote Scores for words NOT in MPQA Lexicon')
plt.xlim([0,1])
plt.ylim([0,12])
n, bins, patches = plt.hist(df_merged_all_notin_mpqa['SUBJSCORE'], 20, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.0)
plt.grid(linestyle='dashed', linewidth=0.5)
plt.grid(True)
plt.show()

print('\n')

#inspecting the distribution
#calculate chi square
notin_mpqa_chisquare = stats.normaltest(df_merged_all_notin_mpqa["SUBJSCORE"])
print('Test for normality (non-MPQA):')
print(notin_mpqa_chisquare)

#calculate kurtosis (flatness/peakiness)
notin_mpqa_kurtosis = stats.kurtosistest(df_merged_all_notin_mpqa["SUBJSCORE"])
print('Kurtosis test (non-MPQA):')
print(notin_mpqa_kurtosis)

#calculate skewness (slant left/right)
notin_mpqa_skew = stats.skewtest(df_merged_all_notin_mpqa["SUBJSCORE"])
print('Skew test (non-MPQA):')
print(notin_mpqa_skew)

#probablity plot
fig = plt.figure()
stats.probplot(df_merged_all_notin_mpqa["SUBJSCORE"], plot=plt, rvalue=True)
plt.show()


print('\n################ Comparing distribution of MPQA strongsubj vs MPQA weaksubj vs non-MPQA ################\n')

#Comparison between MPQA strongsubj and MPQA weaksubj

print('T-test for MPQA strongsubj vs MPQA weaksubj (Group 1 vs 2):')
ttest_all = ttest_ind(strong_merged_all_with_mpqa['SUBJSCORE'], weak_merged_all_with_mpqa['SUBJSCORE'], equal_var = False)
print(ttest_all)

print('Kolmogorov-Smirnov statistic for MPQA strongsubj vs MPQA weaksubj (Group 1 vs 2):')
ks_test = stats.ks_2samp(strong_merged_all_with_mpqa['SUBJSCORE'], weak_merged_all_with_mpqa['SUBJSCORE'])
print(ks_test)

print('Effect size (cohens_d) for MPQA strongsubj vs MPQA weaksubj (Group 1 vs 2):')
c0 = strong_merged_all_with_mpqa["SUBJSCORE"]
c1 = weak_merged_all_with_mpqa["SUBJSCORE"]
cohens_d = (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
print(cohens_d)

# returns the standard normal cumulative distribution function
# % of the treatment group will be above the mean of the control group
normsdist = norm.cdf(cohens_d)
print('Cohens U3:')
print(normsdist)

# % of the two groups will overlap
overlapping_coefficient = 2*norm.cdf(-abs(cohens_d)/2)
print('Overlapping coefficient:')
print(overlapping_coefficient)

# % chance that a person picked at random from the treatment group will have a higher score 
# than a person picked at random from the control group (probability of superiority)
# common language effect size
common_language_ES = norm.cdf(cohens_d/sqrt(2))
print('Common language effect size:')
print(common_language_ES)

print('\n\n')


#Comparison between MPQA strongsubj and non-MPQA

print('T-test for  MPQA strongsubj vs non-MPQA (Group 1 vs 3):')
ttest_all = ttest_ind(strong_merged_all_with_mpqa['SUBJSCORE'], df_merged_all_notin_mpqa['SUBJSCORE'], equal_var = False)
print(ttest_all)

print('Kolmogorov-Smirnov statistic for  MPQA strongsubj vs non-MPQA (Group 1 vs 3):')
ks_test = stats.ks_2samp(strong_merged_all_with_mpqa['SUBJSCORE'], df_merged_all_notin_mpqa['SUBJSCORE'])
print(ks_test)

print('Effect size (cohens_d) for MPQA strongsubj vs non-MPQA (Group 1 vs 3):')
c0 = strong_merged_all_with_mpqa["SUBJSCORE"]
c1 = df_merged_all_notin_mpqa["SUBJSCORE"]
cohens_d = (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
print(cohens_d)

# returns the standard normal cumulative distribution function
# % of the treatment group will be above the mean of the control group
normsdist = norm.cdf(cohens_d)
print('Cohens U3:')
print(normsdist)

# % of the two groups will overlap
overlapping_coefficient = 2*norm.cdf(-abs(cohens_d)/2)
print('Overlapping coefficient:')
print(overlapping_coefficient)

# % chance that a person picked at random from the treatment group will have a higher score 
# than a person picked at random from the control group (probability of superiority)
# common language effect size
common_language_ES = norm.cdf(cohens_d/sqrt(2))
print('Common language effect size:')
print(common_language_ES)

print('\n\n')


#Comparison between MPQA weaksubj and non-MPQA

print('T-test for  MPQA weaksubj vs non-MPQA (Group 2 vs 3):')
ttest_all = ttest_ind(weak_merged_all_with_mpqa['SUBJSCORE'], df_merged_all_notin_mpqa['SUBJSCORE'], equal_var = False)
print(ttest_all)

print('Kolmogorov-Smirnov statistic for  MPQA weaksubj vs non-MPQA (Group 2 vs 3):')
ks_test = stats.ks_2samp(weak_merged_all_with_mpqa['SUBJSCORE'], df_merged_all_notin_mpqa['SUBJSCORE'])
print(ks_test)

print('Effect size (cohens_d) for MPQA weaksubj vs non-MPQA (Group 2 vs 3):')
c0 = weak_merged_all_with_mpqa["SUBJSCORE"]
c1 = df_merged_all_notin_mpqa["SUBJSCORE"]
cohens_d = (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
print(cohens_d)

# returns the standard normal cumulative distribution function
# % of the treatment group will be above the mean of the control group
normsdist = norm.cdf(cohens_d)
print('Cohens U3:')
print(normsdist)

# % of the two groups will overlap
overlapping_coefficient = 2*norm.cdf(-abs(cohens_d)/2)
print('Overlapping coefficient:')
print(overlapping_coefficient)

# % chance that a person picked at random from the treatment group will have a higher score 
# than a person picked at random from the control group (probability of superiority)
# common language effect size
common_language_ES = norm.cdf(cohens_d/sqrt(2))
print('Common language effect size:')
print(common_language_ES)


plt.title('Histogram of Quote Scores for strongsubj vs weaksubj')
plt.hist(strong_merged_all_with_mpqa['SUBJSCORE'], 20, alpha=0.5, label='Group 1')
plt.hist(weak_merged_all_with_mpqa['SUBJSCORE'], 20, alpha=0.5, label='Group 2')
#plt.hist(df_merged_all_notin_mpqa['SUBJSCORE'], 20, alpha=0.5, label='Group 3')
plt.legend(loc='upper right')
#plt.gca().set_yscale("log")
plt.grid(True)
plt.show()



print('\n############################### SENTIWORDNET ###############################\n')

sentiwordnet_input = pd.read_csv('SentiWordNet.csv', sep=";", error_bad_lines=False)
#sentiwordnet_input['WORD'] = sentiwordnet_input['SynsetTerms'].str.split('#', 1).str[0]
sentiwordnet_input['SentiWordScore'] = sentiwordnet_input['PosScore'] + sentiwordnet_input['NegScore']
print(sentiwordnet_input.head())

#processsing the input to aggregate synsets and their respective scores

#extract only the word column into a new dataframe
sentiwordsplit = pd.DataFrame()
sentiwordnet_input = sentiwordnet_input.reset_index(drop=True)
sentiwordsplit = sentiwordsplit.reset_index(drop=True)
sentiwordsplit['SynsetTerms'] = sentiwordnet_input['SynsetTerms']
print(sentiwordsplit.head())

#separate words within the word column by hash delimiter
sentiwordhash = sentiwordsplit['SynsetTerms'].str.split('#', expand=True)
#remove leading numbers in words
sentiwordhash = sentiwordhash.applymap(lambda x: x.lstrip(digits) if x is not None else None)
sentiwordlist = sentiwordhash.values.tolist()
sentiscorelist = sentiwordnet_input['SentiWordScore'].tolist()

#clean up the list
for row_list in sentiwordlist:
    row_list[:] = [word for word in row_list if (word is not None)]
    row_list[:] = [word for word in row_list if (len(word) > 0)]


sentiscorelist_new = []

for i in range(0, len(sentiwordlist)):
#for i in sentiscorelist:
    #looprange = len(sentiwordlist[i]) + 1
    for k in range(0, len(sentiwordlist[i])):       # if len(sentiwordlist[i] = 1, range(0,2) will loop just once
        sentiscorelist_new.append(sentiscorelist[i])

#print(sentiscorelist_new)
sentiwordlist_new = [y for x in sentiwordlist for y in x]
#print(sentiwordlist_new)

df_sentiword_new = pd.DataFrame({'WORD':sentiwordlist_new, 'SentiWordScore':sentiscorelist_new})
df_sentiword_new['WORD'] = df_sentiword_new['WORD'].str.strip()
df_sentiword_new = df_sentiword_new.groupby('WORD').mean().reset_index()


#compare words against NLTK English dictionary to ensure word is English
df_sentiword_new = df_sentiword_new[df_sentiword_new['WORD'].isin(words.words())]


#histogram of "SentiWordScore"
plt.title('Histogram of SENTIWORDSCORE in SentiWordNet')
n, bins, patches = plt.hist(df_sentiword_new['SentiWordScore'], 20, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.0)
plt.grid(linestyle='dashed', linewidth=0.5)
plt.grid(True)
plt.gca().set_yscale("log")
plt.show()

 
#common words between SentiWordNet and Wikipedia lexicon
df_merged_all_with_sentiwordnet = df_sentiword_new.merge(data_frequency_threshold, how='outer')
df_merged_all_with_sentiwordnet = df_merged_all_with_sentiwordnet[pd.notnull(df_merged_all_with_sentiwordnet['SUBJSCORE'])]
df_merged_all_with_sentiwordnet = df_merged_all_with_sentiwordnet[pd.notnull(df_merged_all_with_sentiwordnet['SentiWordScore'])]
#df_merged_all_with_sentiwordnet = df_merged_all_with_sentiwordnet[(df_merged_all_with_sentiwordnet['FREQUENCY_QUOTE'] >= 5) | (df_merged_all_with_sentiwordnet['FREQUENCY_NORMAL'] >= 5)]
df_merged_all_with_sentiwordnet = df_merged_all_with_sentiwordnet.sort_values(by = ['SUBJSCORE', 'FREQUENCY_QUOTE'], ascending=[False, False])
print(df_merged_all_with_sentiwordnet.head())

print('\n')

#calculate median of SentiWordScore
sentiwordnet_score_median = df_merged_all_with_sentiwordnet["SentiWordScore"].median()
print('Median SentiWordScore:')
print(sentiwordnet_score_median)

#calculate mean of Quote Score
sentiwordnet_wiki_median = df_merged_all_with_sentiwordnet["SUBJSCORE"].median()
print('Median SUBJSCORE:')
print(sentiwordnet_wiki_median)

#calculate mean of SentiWordScore
sentiwordnet_score_mean = df_merged_all_with_sentiwordnet["SentiWordScore"].mean()
print('Mean SentiWordScore:')
print(sentiwordnet_score_mean)

#calculate mean of Quote Score
sentiwordnet_wiki_mean = df_merged_all_with_sentiwordnet["SUBJSCORE"].mean()
print('Mean SUBJSCORE:')
print(sentiwordnet_wiki_mean)

#calculate standard deviation of SentiWordScore
sentiwordnet_score_std = df_merged_all_with_sentiwordnet["SentiWordScore"].std()
print('Std_dev SentiWordScore:')
print(sentiwordnet_score_std)

#calculate standard deviation of Quote Score
sentiwordnet_wiki_std = df_merged_all_with_sentiwordnet["SUBJSCORE"].std()
print('Std_dev SUBJSCORE:')
print(sentiwordnet_wiki_std)


plt.title('Histogram of SentiWordScore in SentiWordNet')
n, bins, patches = plt.hist(df_merged_all_with_sentiwordnet['SentiWordScore'], 20, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.0)
plt.grid(linestyle='dashed', linewidth=0.5)
plt.grid(True)
plt.gca().set_yscale("log")
plt.show()


plt.title('Histogram of Quote Score and SentiWordScore')
plt.hist(df_merged_all_with_sentiwordnet['SentiWordScore'], 20, alpha=0.5, label='SentiWordScore')
plt.hist(df_merged_all_with_sentiwordnet['SUBJSCORE'], 20, alpha=0.5, label='Quote Score')
plt.legend(loc='upper right')
plt.gca().set_yscale("log")
plt.grid(True)
plt.show()


plt.title('Quote Score vs SentiWordScore')
plt.plot(df_merged_all_with_sentiwordnet['SUBJSCORE'], df_merged_all_with_sentiwordnet['SentiWordScore'], '.')
plt.xlabel('Quote Score')
plt.ylabel('SentiWordScore')
plt.grid(True)
plt.show()


plt.title('Quote Score vs SentiWordScore (log scale)')
plt.plot(df_merged_all_with_sentiwordnet['SUBJSCORE'], df_merged_all_with_sentiwordnet['SentiWordScore'], '.')
plt.xlabel('Quote Score')
plt.ylabel('SentiWordScore')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.grid(True)
plt.show()

print('Skew test of quote score:')
print(stats.skewtest(df_merged_all_with_sentiwordnet["SUBJSCORE"]))

print('Skew test of SentiWordScore:')
print(stats.skewtest(df_merged_all_with_sentiwordnet["SentiWordScore"]))

#Kendall's tau rank correlation
tau, p_value = stats.kendalltau(df_merged_all_with_sentiwordnet['SentiWordScore'], df_merged_all_with_sentiwordnet['SUBJSCORE'])
print('Tau value:')
print(tau)
print('p-value:')
print(p_value)

#Spearman's rank correlation
rho, p_value = stats.spearmanr(df_merged_all_with_sentiwordnet['SentiWordScore'], df_merged_all_with_sentiwordnet['SUBJSCORE'])
print('Rho value:')
print(rho)
print('p-value:')
print(p_value)


#comparing SentiWordNet and Quote Score distributions

print('T-test for SentiWordNet vs Quote Scores:')
ttest_all = ttest_ind(df_merged_all_with_sentiwordnet['SentiWordScore'], df_merged_all_with_sentiwordnet['SUBJSCORE'], equal_var = False)
print(ttest_all)

print('Kolmogorov-Smirnov statistic for SentiWordNet vs Quote Scores:')
ks_test = stats.ks_2samp(df_merged_all_with_sentiwordnet['SentiWordScore'], df_merged_all_with_sentiwordnet['SUBJSCORE'])
print(ks_test)

print('Mann–Whitney U test:')
pairedt_test = stats.mannwhitneyu(df_merged_all_with_sentiwordnet['SUBJSCORE'], df_merged_all_with_sentiwordnet['SentiWordScore'])
print(pairedt_test)

print('Wilcoxon signed-rank test:')
signedrank_test = stats.wilcoxon(df_merged_all_with_sentiwordnet['SUBJSCORE'], df_merged_all_with_sentiwordnet['SentiWordScore'])
print(signedrank_test)

n1 = len(df_merged_all_with_sentiwordnet['SUBJSCORE'])
n2 = len(df_merged_all_with_sentiwordnet['SentiWordScore'])

print('\nWilcoxon CHECK \n')

mean_wilcoxon = n1*(n1+1)/4
print(mean_wilcoxon)
stddev_wilcoxon = sqrt((n1*(n1+1)*(2*n1+1))/24)
print(stddev_wilcoxon)
Z_wilcoxon = (signedrank_test.statistic - mean_wilcoxon)/stddev_wilcoxon
print(Z_wilcoxon)

print('\nMann–Whitney CHECK \n')

mean_Utest = 0.5*n1*n2
print('Mean U test value:')
print(mean_Utest)

stddev_Utest = sqrt(((n1*n2)*(n1+n2+1))/12)
print('Utest std dev value:')
print(stddev_Utest)


Z_Utest = 1.96

Ucritvalue = mean_Utest - (Z_Utest*stddev_Utest) - 0.5
print('U critical value:')
print(Ucritvalue)


print('Effect size (cohens_d) for Quote Score vs SentiWordScore:')
c0 = df_merged_all_with_sentiwordnet["SentiWordScore"]
c1 = df_merged_all_with_sentiwordnet["SUBJSCORE"]
cohens_d = (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
print(cohens_d)

# returns the standard normal cumulative distribution function
# % of the treatment group will be above the mean of the control group
normsdist = norm.cdf(cohens_d)
print('Cohens U3:')
print(normsdist)

# % of the two groups will overlap
overlapping_coefficient = 2*norm.cdf(-abs(cohens_d)/2)
print('Overlapping coefficient:')
print(overlapping_coefficient)

# % chance that a person picked at random from the treatment group will have a higher score 
# than a person picked at random from the control group (probability of superiority)
# common language effect size
common_language_ES = norm.cdf(cohens_d/sqrt(2))
print('Common language effect size:')
print(common_language_ES)


#check for significant disagreement - FOR EXPERIMENTAL PURPOSE ONLY
#SentiWordScore higher than mean and Quote Score lower than mean
#SentiWordScore lower than mean and Quote Score higher than mean
#difference between scores greater than 0.1
df_sentiword_disagreement = df_merged_all_with_sentiwordnet[(((df_merged_all_with_sentiwordnet['SentiWordScore'] > sentiwordnet_score_mean) & (df_merged_all_with_sentiwordnet['SUBJSCORE'] < sentiwordnet_wiki_mean)) |
                                                             ((df_merged_all_with_sentiwordnet['SentiWordScore'] < sentiwordnet_score_mean) & (df_merged_all_with_sentiwordnet['SUBJSCORE'] > sentiwordnet_wiki_mean))) &
                                                            (abs(df_merged_all_with_sentiwordnet['SentiWordScore'] - df_merged_all_with_sentiwordnet['SUBJSCORE']) > 0.5)]


cols = ['WORD','SUBJSCORE']
                                                                 
#==============================================================================
# medicalwords =  pd.read_csv('medical_wordlist.csv', names = ["WORD"])  
# 
# df_sentiword_medical = pd.merge(df_merged_all_with_sentiwordnet, medicalwords, how='inner', on=['WORD'])                 
# #df_sentiword_medical = pd.merge(df_sentiword_disagreement, medicalwords, how='inner', on=['WORD'])  ###########
# #get words in my lexicon which are NOT in medical lexicon
# #get copies where the indices are the columns of interest
# 
# f2 = df_merged_all_with_sentiwordnet.set_index(cols)
# other2 = df_sentiword_medical.set_index(cols)
# #look for index overlap
# df_merged_all_sentiwordnet_agreement = df_merged_all_with_sentiwordnet[~df2.index.isin(other2.index)]
#==============================================================================


df2 = df_sentiword_disagreement.set_index(cols)
df_merged_all_sentiwordnet_agreement = df_sentiword_disagreement[~df2.index.isin(other2.index)]
print(df_merged_all_sentiwordnet_agreement.head())


print('Skew test of quote score:')
print(stats.skewtest(df_merged_all_sentiwordnet_agreement["SUBJSCORE"]))

print('Skew test of SentiWordScore:')
print(stats.skewtest(df_merged_all_sentiwordnet_agreement["SentiWordScore"]))

#T-test - MPQA strongsubj vs MPQA weaksubj (Group 1 vs 2)
print('T-test for SentiWordNet vs Quote Scores:')
ttest_all = ttest_ind(df_merged_all_sentiwordnet_agreement['SentiWordScore'], df_merged_all_sentiwordnet_agreement['SUBJSCORE'], equal_var = False)
print(ttest_all)

print('Kolmogorov-Smirnov statistic for SentiWordNet vs Quote Scores:')
ks_test = stats.ks_2samp(df_merged_all_sentiwordnet_agreement['SentiWordScore'], df_merged_all_sentiwordnet_agreement['SUBJSCORE'])
print(ks_test)

print('Mann–Whitney U test:')
pairedt_test = stats.mannwhitneyu(df_merged_all_sentiwordnet_agreement['SUBJSCORE'], df_merged_all_sentiwordnet_agreement['SentiWordScore'])
print(pairedt_test)


n1 = len(df_merged_all_sentiwordnet_agreement['SUBJSCORE'])
n2 = len(df_merged_all_sentiwordnet_agreement['SentiWordScore'])

print('\nWilcoxon CHECK \n')

mean_wilcoxon = n1*(n1+1)/4
print(mean_wilcoxon)
stddev_wilcoxon = sqrt((n1*(n1+1)*(2*n1+1))/24)
print(stddev_wilcoxon)
Z_wilcoxon = (signedrank_test.statistic - mean_wilcoxon)/stddev_wilcoxon
print(Z_wilcoxon)

print('\nMann–Whitney CHECK \n')


mean_Utest = 0.5*n1*n2
print('Mean U test value:')
print(mean_Utest)

stddev_Utest = sqrt(((n1*n2)*(n1+n2+1))/12)
print('Utest std dev value:')
print(stddev_Utest)

Z_Utest = 1.96

Ucritvalue = mean_Utest - (Z_Utest*stddev_Utest) - 0.5
print('U critical value:')
print(Ucritvalue)


print('Effect size (cohens_d) for Quote Score vs SentiWordScore:')
c0 = df_merged_all_sentiwordnet_agreement["SentiWordScore"]
c1 = df_merged_all_sentiwordnet_agreement["SUBJSCORE"]
cohens_d = (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
print(cohens_d)


#adjective_words = pd.read_csv('adjective_words.csv', header=None, encoding="UTF-8", errors='ignore')
#adjective_words = codecs.open('adjective_words.csv', 'r', errors = 'ignore')



#==================================================== Remaining Words ======================================================================

#get words in my lexicon which are NOT in MPQA lexicon and SentiWordNet
cols = ['WORD','SUBJSCORE']
#get copies where the indices are the columns of interest
df2 = df_merged_all_notin_mpqa.set_index(cols)
other2 = df_merged_all_with_sentiwordnet.set_index(cols)
#look for index overlap
df_notin_mpqa_sentiwordnet = df_merged_all_notin_mpqa[~df2.index.isin(other2.index)]
print(df_notin_mpqa_sentiwordnet.head())


#histogram of words in my lexicon which are NOT in MPQA lexicon and SentiWordNet
plt.title('Histogram of Words NOT in MPQA and SentiWordNet')
n, bins, patches = plt.hist(df_notin_mpqa_sentiwordnet['SUBJSCORE'], 20, normed=1, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.0)
plt.grid(linestyle='dashed', linewidth=0.5)
plt.grid(True)
plt.show()


#merge MPQA and SentiWordNet
df_sentiwordnet_mpqa = mpqa_lexicon.merge(df_sentiword_new, how='outer')
df_sentiwordnet_mpqa = df_sentiwordnet_mpqa[pd.notnull(df_sentiwordnet_mpqa['SUBJTYPE'])]
df_sentiwordnet_mpqa = df_sentiwordnet_mpqa[pd.notnull(df_sentiwordnet_mpqa['SentiWordScore'])]
print(df_sentiwordnet_mpqa.head())


#merge ALL
df_merge_all3 = df_sentiwordnet_mpqa.merge(data_frequency_threshold, how='outer')
df_merge_all3 = df_merge_all3[pd.notnull(df_merge_all3['SUBJSCORE'])]
df_merge_all3 = df_merge_all3[pd.notnull(df_merge_all3['SUBJTYPE'])]
df_merge_all3 = df_merge_all3[pd.notnull(df_merge_all3['SentiWordScore'])]
print(df_merge_all3.head())


df_merge_all3_agreement = df_merge_all3[(((df_merge_all3['SentiWordScore'] > sentiwordnet_score_mean) & (df_merge_all3['SUBJSCORE'] > sentiwordnet_wiki_mean) & df_merge_all3['SUBJTYPE'].str.match('strongsubj')) |
                                         ((df_merge_all3['SentiWordScore'] < sentiwordnet_score_mean) & (df_merge_all3['SUBJSCORE'] < sentiwordnet_wiki_mean) & df_merge_all3['SUBJTYPE'].str.match('weaksubj'))) &
                                         (abs(df_merge_all3['SentiWordScore'] - df_merge_all3['SUBJSCORE']) < 0.05)]


#============================================== Sampling for Survey ==========================================================  
                                       
df_merge_all3_agreement_sample = df_merge_all3_agreement.sample(20)
print('\nSample Control Group:')  
print(df_merge_all3_agreement_sample)


df_notin_mpqa_sentiwordnet = df_notin_mpqa_sentiwordnet.sort_values(by='SUBJSCORE', ascending=False)

df_bucket_low = df_notin_mpqa_sentiwordnet[df_notin_mpqa_sentiwordnet['SUBJSCORE'] < df_notin_mpqa_sentiwordnet['SUBJSCORE'].mean()]
print('\nSample Low Quote Score Bucket:')  
print(df_bucket_low.sample(20))

df_bucket_high = df_notin_mpqa_sentiwordnet.head(500)
print('\nSample High Quote Score Bucket:')  
print(df_bucket_high.sample(30))


df_lowhigh = df_bucket_low.merge(df_bucket_high, how='outer')
df2 = df_notin_mpqa_sentiwordnet.set_index(cols)
other2 = df_lowhigh.set_index(cols)
df_bucket_medium = df_notin_mpqa_sentiwordnet[~df2.index.isin(other2.index)]
print('\nSample Medium Quote Score Bucket:')  
print(df_bucket_medium.sample(30))                                              