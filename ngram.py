from nltk import bigrams
from nltk.tokenize import sent_tokenize
import text_preprocessor as tp
import math
import operator
import re
import io
import sys


def add_UNK_symbol(dic):
    dic['<UNK>'] = 0
    for key in dic:
        if dic[key] < 5:
            dic['<UNK>'] += 1
    return dic


def add_stop_symbol(word_list):
    for index, word in enumerate(word_list):
        if re.match(r"[.!?]+(\")*", word):
            word_list[index] = '<s>'
            word_list.insert(index, '</s>')
    word_list.insert(0, word_list.pop())
    return word_list

if __name__ == "__main__":

    train_in = sys.argv[1]
    test_in = sys.argv[2]
    output_file = sys.argv[3]

    train_file = io.open(train_in, "r", encoding="cp1250")
    sys.stdout = open(output_file, 'w')

    train_text = train_file.read()
    number_of_sentences = tp.count_sentences(train_text)
    word_list = tp.word_tokenize(train_text)
    token_dict = tp.type_token_dict(
        word_list, 'token')  # token count dictionary
    total_count = tp.total_token_count(
        token_dict) + number_of_sentences  # Added number of sentences
    token_dict = add_UNK_symbol(token_dict)  # Add the <UNK> symbol

    # Adding stop symbols '<s>', '</s>' to word list
    word_list = add_stop_symbol(word_list)

    bigram_train_dict = {}
    # bigrams_list = list(bigrams(word_list))
    for bigram in bigrams(word_list):
        if bigram not in bigram_train_dict:
            bigram_train_dict[bigram] = 1
        else:
            bigram_train_dict[bigram] += 1

    test_file = io.open(test_in, "r", encoding="cp1250")
    test_text = test_file.read()
    test_sentences = sent_tokenize(test_text)
    test_sentence_count = len(test_sentences)

    def unigram_prob_sentence(sentence):
        sentence_prob = 0

        for word in sentence:
            if word not in token_dict:
                numerator = token_dict['<UNK>']
            else:
                numerator = token_dict[word]
            unigram_prob = numerator / float(total_count)
            sentence_prob += math.log(unigram_prob)

        # This was added to account for stop symbol!
        sentence_prob += math.log(number_of_sentences / float(total_count))
        return math.exp(sentence_prob)

    def bigram_prob_sentence(sentence):
        sentence_prob = 0
        denominator = 0

        for bigram in bigrams(sentence):

            if bigram not in bigram_train_dict:
                test_word = bigram[1]
                if test_word not in token_dict:
                    test_word = '<UNK>'

                # HERE is where we need to apply the unknown word
                numerator = token_dict[test_word]
                denominator = total_count

            elif '<s>' == bigram[0] or '</s>' == bigram[0]:
                numerator = bigram_train_dict[bigram]
                denominator = number_of_sentences

            else:
                numerator = bigram_train_dict[bigram]
                denominator = token_dict[bigram[0]]

            bigram_prob = numerator / float(denominator)
            sentence_prob += math.log(bigram_prob)

        return math.exp(sentence_prob)

    total_unigram_probs = 0.0
    total_bigram_probs = 0.0

    for i, sentence in enumerate(test_sentences):
        try:
            print "Sentence ", i + 1, ': ', sentence
        except UnicodeError:
            print "Sentence ", i + 1, ': '

        sentence = tp.convert_contractions(tp.word_tokenize(sentence))
        unigram_prob = unigram_prob_sentence(sentence)
        total_unigram_probs += unigram_prob
        print " - unigram [Prob] ", unigram_prob
        sentence = add_stop_symbol(sentence)
        bigram_prob = bigram_prob_sentence(sentence)
        total_bigram_probs += bigram_prob
        print ' - bigram  [Prob] ', bigram_prob
        print

    avg_unigram = total_unigram_probs / test_sentence_count
    avg_bigram = total_bigram_probs / test_sentence_count
    print '====================='
    print ' * Probability:'
    print ' - Average Unigram Probability:', avg_unigram
    print ' - Average Bigram Probability:', avg_bigram
