# ngram-model

This program uses a training text to generate probabilites for a test text. 

Usage: $python  ngram.py <train-text> <test-text> <output file>

A brief primer on ngram probabilities...

Given a tiny train text:

I am Sam.
Sam I am.
I do not like green eggs and ham. 

The bigram model would be generated like so:

(I, am) (am, Sam) (Sam, '.')
(Sam, I) (I, am) (am, '.')
(I, do) (do, not) (not, like) (like, green) (green, eggs) (eggs, and) (and, ham) (ham, '.')

Then we can ask the following, "Given the word "I", what is the probability we'll see the word "am" ?" 

We can use a naive Markov assumption to say that the probability of word, only depends on the previous word i.e.

P(am|I) = Count(Bigram(I,am)) / Count(Word(I))

The probability of the sentence is simply multiplying the probabilities of all the respecitive bigrams. 

Note: I used Log probabilites and backoff smoothing in my model. 



 

