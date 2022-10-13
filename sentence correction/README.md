# Sentence Generation and Correction
using markov and hidden markov models to generate sentences and correct the given sentence

# Setup

To run this, all one needs is Python 3.4 or above and `pip3`.

To install dependencies, simply run:

```bash
pip3 install -r requirements.txt
```

# Details

This assignment has two parts:

## Sentence Generation

Sentence are randomly generated using Markov models defined in [`data`](data).

To run, simply execute:

```bash
python3 generator.py
```

More detailed output can be displayed using th e `-v` or `--verbose` flags.

## Sentence Correction

Sentence are corrected using a Hidden Markov Model (HMM) and Viterbi's
algorithm. The intput is read from `stdin`, and the output is the most likely
sentence based on a first-order Markov chain and the Levenshtein distance.

To run, simply execute:

```bash
python3 corrector.py
```

Input can also be piped in as follows:

```bash
echo "How are yuo" | python3 corrector.py
# How are you
```

More detailed output can be displayed using th e `-v` or `--verbose` flags.
