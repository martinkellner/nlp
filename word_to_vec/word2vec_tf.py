import numpy as np
import tensorflow as tf

from keras.preprocessing import (
    sequence,
    text
)
from keras.utils import np_utils

from nltk import sent_tokenize, word_tokenize, pos_tag


def get_tokenized_sentences(corpus):
    sentences = []
    print(sent_tokenize(corpus))
    for sentence in sent_tokenize(corpus):
        sentences.append(sentence)

    # Covnert text to training samples (e.g: (there, are), (are, many))
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts([corpus])

    word2id = tokenizer.word_index
    word2id['PAD'] = 0
    id2word = {v: k for k, v in word2id.items()}

    vocab_size = len(id2word)
    print("The total number of unique words: {}".format(vocab_size))
    wids = [[word2id[w]
             for w in text.text_to_word_sequence(sen)] for sen in sentences]
    # Generate training samples
    return wids, id2word, word2id


def generate_context_word_pair(wids, win_size, vocab_size):
    context_lenght = win_size*2
    for words in wids:
        sentence_lenght = len(words)
        for index, word in enumerate(words):
            context_word = []
            label_word = []
            start = index - win_size if 0 <= index - win_size else 0
            end = ((index + win_size + 1)
                   if (index + win_size + 1) < sentence_lenght
                   else sentence_lenght - 1)

            context_word.append([words[i]
                                 for i in range(start, end)
                                 if i != index])
            label_word.append(word)

            pad_sq = sequence.pad_sequences(context_word,
                                            maxlen=context_lenght)
            x = [np_utils.to_categorical([w], vocab_size) for w in pad_sq]
            y = np_utils.to_categorical(label_word, vocab_size)

            yield x, y

def get_training_x_and_y(inp_text, win_size):
    wids, id2word, word2id = get_tokenized_sentences(inp_text)
    x_train, y_train = [], []
    for x, y in generate_context_word_pair(wids, win_size, len(word2id)):
        for context_word in x[0][0]:
            x_train.append(context_word)
            y_train.append(y[0])
    print(np.array(x_train).shape, np.array(y_train).shape)
    return x_train, y_train, id2word, word2id

def train_using_tensorflow(x_train, y_train, win_size, vocab_size, debug=True):
    x_plc_holder = tf.placeholder(tf.float32,
                                  shape=(None, vocab_size))
    y_plc_holder = tf.placeholder(tf.float32,
                                  shape=(None, vocab_size))

    EMBEDDING_DIM = 5
    W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
    b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))

    # Embedding
    hidden_representation = tf.add(tf.matmul(x_plc_holder, W1), b1)

    W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
    b2 = tf.Variable(tf.random_normal([vocab_size]))

    prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2),
                                      b2))

    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)

    cross_endropy_loss = tf.reduce_mean(
        -tf.reduce_sum(y_plc_holder *
                       tf.log(prediction),
                       reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(
        cross_endropy_loss)

    n_iters = 5000
    for ep in range(n_iters):
        sess.run(train_step,
                 feed_dict={x_plc_holder: x_train, y_plc_holder: y_train})
        if debug:
            print("Epoch :{},  Loss: {}"
                .format(ep + 1, sess.run(cross_endropy_loss,
                                        feed_dict={x_plc_holder: x_train,
                                                    y_plc_holder: y_train})
                        )
                )
    vectors = sess.run(W1 + b1)
    return vectors

def distance(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(vectors, word2id, id2word, word):
    word_index = word2id[word]
    word_vector = vectors[word_index]
    distances = [distance(word_vector, vectors[w2])
                 for w2 in range(len(word2id))
                 if w2 != word_index]

    sorted_index = np.argsort(np.array(distances))
    print(sorted_index)
    closest_word = [id2word[i] for i in sorted_index]
    print("Closest words for the given word: '{}' -> {}"
          .format(word, closest_word[:5]))


# inp_text = "There are many variations of passages of Lorem Ipsum available," \
#     "but the majority have suffered alteration in some form, by injected hu" \
#     "mour, or randomised words which don't look even slightly believable. I" \
#     "f you are going to use a passage of Lorem Ipsum, you need to be sure t" \
#     "here isn't anythofing embarrassing hidden in the middle of text. All t" \
#     "he Lorem Ipsum generators on the Internet tend to repeat predefined ch" \
#     "unks as necessary, making this the first true generator on the Interne" \
#     "t. It uses a dictionary of over 200 Latin words, combined with a handf" \
#     "ul of model sentence structures, to generate Lorem Ipsum which looks r" \
#     "easonable. The generated Lorem Ipsum is therefore always free from rep" \
#     "etition, injected humour, or non-characteristic words etc."

data = ""
with open("bible.txt", 'r') as file:
    arr = file.readlines()
    print(arr)
    data = ''.join(arr)
print(type(data))
window_size = 2
x_train, y_train, id2word, word2id = get_training_x_and_y(data,
                                                          win_size=window_size)

vectors = train_using_tensorflow(x_train, y_train, window_size, len(word2id))
find_closest(vectors, word2id, id2word, id2word[5])
