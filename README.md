# Topic Recognizer
The algorithm consists of two layers:
1. A text normalizer using `mystem` and `nltk`
2. A pipeline of `TF-IDF` and `Latent Dirichlet Allocation`

The model recognizes 10000 most frequently used words and is capable of identifying 300 topics

The model was trained on a data set consisting of 100000 articles and ~83200000 words.

The training took 3 hours and 42 minutes on i9-9990 CPU + 48Gb RAM
