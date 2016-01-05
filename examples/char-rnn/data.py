# pls get linux_input.txt from http://cs.stanford.edu/people/karpathy/char-rnn/
data = open('linux_input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
with open('vocab.txt', 'w') as fd:
  fd.write("".join(chars))
  fd.flush()
