#encoding: utf-8
import os
import codecs
import numpy as np
# def get_glove_words(glove_file_path):
#     if not os.path.exists(glove_file_path):
#         print(glove_file_path + ' not exists!')
#         return []
#     words_list = []
#     with open(glove_file_path, 'r', encoding="utf-8") as fo:
#         for line in fo:
#             word = line.encode().decode('utf-8').strip().split(' ')[0]
#             #print(word)
#             words_list.append(word)
#     return words_list

# words_list = get_glove_words('glove.840B.300d.txt')
# print(len(words_list))
# with open('glove_words.txt', 'w', encoding="utf-8") as fo:
#     for w in words_list:
#         fo.write(w+'\n')

def convert_to_binary(embedding_path):
    """
    Here, it takes path to embedding text file provided by glove.
    :param embedding_path: takes path of the embedding which is in text format or any format other than binary.
    :return: a binary file of the given embeddings which takes a lot less time to load.
    """
    f = codecs.open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []
    with codecs.open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            if count == 0:
                pass
            else:
                splitlines = line.split()
                vocab_write.write(splitlines[0].strip())
                vocab_write.write("\n")
                wv.append([float(val) for val in splitlines[1:]])
            count += 1
    np.save(embedding_path + ".npy", np.array(wv))

convert_to_binary('/home/tanvi/Downloads/archive1/glove.840B.300d')