/*
 * This file include code from rnnlmlib-0.4 whose licence is as follows:
Copyright (c) 2010-2012 Tomas Mikolov
Copyright (c) 2013 Cantab Research Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//
// This code creates DataShard for RNNLM dataset.
// The RNNLM dataset could be downloaded at
//    http://www.rnnlm.org/
//
// Usage:
//    create_shard.bin -train train_file -class_size [-debug] [-valid valid_file] [-test test_file]

#include "utils/data_shard.h"
#include "utils/common.h"
#include "proto/common.pb.h"
#include "singa.h"
#include "rnnlm.pb.h"

#define MAX_STRING 100

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>

using namespace std;
using singa::DataShard;

struct vocab_word {
    int cn;
    char word[MAX_STRING];
    int class_index;
};

struct vocab_word *vocab;
int vocab_max_size;
int vocab_size;
int *vocab_hash;
int vocab_hash_size;
int debug_mode;
int old_classes;
int *class_start;
int *class_end;
int class_size;

char train_file[MAX_STRING];
char valid_file[MAX_STRING];
char test_file[MAX_STRING];

int valid_mode;
int test_mode;

unsigned int getWordHash(char *word) {
    unsigned int hash, a;

    hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 237 + word[a];
    hash = hash % vocab_hash_size;

    return hash;
}

int searchVocab(char *word) {
    int a;
    unsigned int hash;

    hash = getWordHash(word);

    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];

    for (a = 0; a < vocab_size; a++) {                //search in vocabulary
        if (!strcmp(word, vocab[a].word)) {
            vocab_hash[hash] = a;
            return a;
        }
    }

    return -1;                            //return OOV if not found
}

int addWordToVocab(char *word) {
    unsigned int hash;

    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;

    if (vocab_size + 2 >= vocab_max_size) {        //reallocate memory if needed
        vocab_max_size += 100;
        vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }

    hash = getWordHash(word);
    vocab_hash[hash] = vocab_size - 1;

    return vocab_size - 1;
}

void readWord(char *word, FILE *fin) {
    int a = 0, ch;

    while (!feof(fin)) {
        ch = fgetc(fin);

        if (ch == 13) continue;

        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }

            if (ch == '\n') {
                strcpy(word, (char *) "</s>");
                return;
            }
            else continue;
        }

        word[a] = char(ch);
        a++;

        if (a >= MAX_STRING) {
            //printf("Too long word found!\n");   //truncate too long words
            a--;
        }
    }
    word[a] = 0;
}

void sortVocab() {
    int a, b, max;
    vocab_word swap;

    for (a = 1; a < vocab_size; a++) {
        max = a;
        for (b = a + 1; b < vocab_size; b++) if (vocab[max].cn < vocab[b].cn) max = b;

        swap = vocab[max];
        vocab[max] = vocab[a];
        vocab[a] = swap;
    }
}

int learnVocabFromTrainFile() {
    char word[MAX_STRING];
    FILE *fin;
    int a, i, train_wcn;

    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

    fin = fopen(train_file, "rb");

    vocab_size = 0;

    addWordToVocab((char *) "</s>");

    train_wcn = 0;
    while (1) {
        readWord(word, fin);
        if (feof(fin)) break;

        train_wcn++;

        i = searchVocab(word);
        if (i == -1) {
            a = addWordToVocab(word);
            vocab[a].cn = 1;
        } else vocab[i].cn++;
    }

    sortVocab();

    if (debug_mode > 0) {
        printf("Vocab size: %d\n", vocab_size);
        printf("Words in train file: %d\n", train_wcn);
    }

    //train_words = train_wcn;

    fclose(fin);
    return 0;
}

int splitClasses() {
    double df, dd;
    int i, a, b;

    df = 0;
    dd = 0;
    a = 0;
    b = 0;

    class_start = (int *) calloc(class_size, sizeof(int));
    memset(class_start, 0x7f, sizeof(int) * class_size);
    class_end = (int *) calloc(class_size, sizeof(int));
    memset(class_end, 0, sizeof(int) * class_size);

    if (old_classes) {    // old classes
        for (i = 0; i < vocab_size; i++) b += vocab[i].cn;
        for (i = 0; i < vocab_size; i++) {
            df += vocab[i].cn / (double) b;
            if (df > 1) df = 1;
            if (df > (a + 1) / (double) class_size) {
                vocab[i].class_index = a;
                if (a < class_size - 1) a++;
            } else {
                vocab[i].class_index = a;
            }
        }
    } else {            // new classes
        for (i = 0; i < vocab_size; i++) b += vocab[i].cn;
        for (i = 0; i < vocab_size; i++) dd += sqrt(vocab[i].cn / (double) b);
        for (i = 0; i < vocab_size; i++) {
            df += sqrt(vocab[i].cn / (double) b) / dd;
            if (df > 1) df = 1;
            if (df > (a + 1) / (double) class_size) {
                vocab[i].class_index = a;
                if (a < class_size - 1) a++;
            } else {
                vocab[i].class_index = a;
            }
        }
    }

    // after dividing classes, update class start and class end information
    for(i = 0; i < vocab_size; i++)  {
        a = vocab[i].class_index;
        class_start[a] = min(i, class_start[a]);
        class_end[a] = max(i + 1, class_end[a]);
    }
    return 0;
}

int init_class() {
    //debug_mode = 1;
    debug_mode = 0;
    vocab_max_size = 100;  // largest length value for each word
    vocab_size = 0;
    vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash_size = 100000000;
    vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
    old_classes = 1;

    // read vocab
    learnVocabFromTrainFile();

    // split classes
    splitClasses();

    return 0;
}

int create_shard(const char *input_file, const char *output_file) {
    DataShard dataShard(output_file, DataShard::kCreate);
    singa::WordRecord wordRecord;

    char word[MAX_STRING], str_buffer[32];
    FILE *fin;
    int a, i;
    fin = fopen(input_file, "rb");
    int wcnt = 0;
    while (1) {
        readWord(word, fin);
        if (feof(fin)) break;
        i = searchVocab(word);
        if (i == -1) {
            if (debug_mode) printf("unknown word [%s] detected!", word);
        } else {
            wordRecord.set_word(string(word));
            wordRecord.set_word_index(i);
            int class_idx = vocab[i].class_index;
            wordRecord.set_class_index(class_idx);
            wordRecord.set_class_start(class_start[class_idx]);
            wordRecord.set_class_end(class_end[class_idx]);
            int length = snprintf(str_buffer, 32, "%05d", wcnt++);
            dataShard.Insert(string(str_buffer, length), wordRecord);
        }
    }

    dataShard.Flush();
    fclose(fin);
    return 0;
}

int argPos(char *str, int argc, char **argv) {
    int a;

    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) return a;

    return -1;
}

int main(int argc, char **argv) {
    int i;
    FILE *f;

    //set debug mode
    i = argPos((char *) "-debug", argc, argv);
    if (i > 0) {
        debug_mode = 1;
        if (debug_mode > 0)
            printf("debug mode: %d\n", debug_mode);
    }

    //search for train file
    i = argPos((char *) "-train", argc, argv);
    if (i > 0) {
        if (i + 1 == argc) {
            printf("ERROR: training data file not specified!\n");
            return 0;
        }

        strcpy(train_file, argv[i + 1]);

        if (debug_mode > 0)
            printf("train file: %s\n", train_file);

        f = fopen(train_file, "rb");
        if (f == NULL) {
            printf("ERROR: training data file not found!\n");
            return 0;
        }
        fclose(f);
    } else {
        printf("ERROR: training data must be set.\n");
    }

    //search for valid file
    i = argPos((char *) "-valid", argc, argv);
    if (i > 0) {
        if (i + 1 == argc) {
            printf("ERROR: validating data file not specified!\n");
            return 0;
        }

        strcpy(valid_file, argv[i + 1]);

        if (debug_mode > 0)
            printf("valid file: %s\n", valid_file);

        f = fopen(valid_file, "rb");
        if (f == NULL) {
            printf("ERROR: validating data file not found!\n");
            return 0;
        }
        fclose(f);
        valid_mode = 1;
    }

    //search for test file
    i = argPos((char *) "-test", argc, argv);
    if (i > 0) {
        if (i + 1 == argc) {
            printf("ERROR: testing data file not specified!\n");
            return 0;
        }

        strcpy(test_file, argv[i + 1]);

        if (debug_mode > 0)
            printf("test file: %s\n", test_file);

        f = fopen(test_file, "rb");
        if (f == NULL) {
            printf("ERROR: testing data file not found!\n");
            return 0;
        }
        fclose(f);
        test_mode = 1;
    }

    //search for class size
    i = argPos((char *) "-class_size", argc, argv);
    if (i > 0) {
        if (i + 1 == argc) {
            printf("ERROR: class size not specified!\n");
            return 0;
        }

        class_size = atoi(argv[i + 1]);

        if (debug_mode > 0)
            printf("class size: %d\n", class_size);
    }
    if (class_size <= 0) {
        printf("ERROR: no or invalid class size received!\n");
        return 0;
    }

    init_class();

    create_shard(train_file, "train_shard");
    if (valid_mode) create_shard(valid_file, "valid_shard");
    if (test_mode) create_shard(test_file, "test_shard");

    return 0;
}
