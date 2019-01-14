import os
import spacy
import itertools
import tqdm
import json
import re
from collections import Counter

def load_data(data_dir, data_file):
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line

spacy_en = spacy.load("en")
def tokenize(line):
    for token in spacy_en(line):
        yield token.text

def write_output(tokens, out_file, out_dir):
    out_path = os.path.join(out_dir, out_file)
    with open(out_path, 'w', encoding='utf-8') as f:
        for token in tokens:
            if token != "\n":
                token += "\n"
            f.write(token)

def send_to(src, *dests):
    """
    src is a generator
    dests is a list of generators that accept send
    """
    for x in src:
        for dest in dests:
            dest.send(x)

def write_counter(counter, out_dir, out_file, delimiter = '\t'):
    """
    write Counter to file with one entry per line
    """
    with open(os.path.join(out_dir, out_file), 'w', encoding = 'utf-8') as f:
        for key in counter:
            f.write("{0}{1}{2}\n", key, delimiter, counter[key])

def write_list(lst, out_dir, out_file):
    """
    write List to file with one entry per line
    """
    with open(os.path.join(out_dir, out_file), 'w', encoding = 'utf-8') as f:
        for line in lst:
            f.write(line + "\n")

def write_json(obj, out_dir, out_file):
    with open(os.path.join(out_dir, out_file), 'w', encoding = 'utf-8') as f:
        json.dump(obj, f)

def run_all(data_dir = "../data", input_file = "microblog2011.txt", out_dir = "../output", **kwargs):
    """
    Run all the code
    subroutines: set<string> that contains list of subroutines to call
    """

    # set of all keyword arguments where the value is True
    flags = set(k for k in kwargs if kwargs[k])

    show_progress = 'show_progress' in flags
    if show_progress:
        line_count = sum(1 for _ in load_data(data_dir, input_file))

    """
    SECTION: A
    Submit a file microblog2011_tokenized.txt with the tokenizer’s output for the whole corpus. 
    Include in your report the output for the first 20 sentences in the corpus.
    """
    def tokenize_dataset(data_stream, out_file):
        token_stream = itertools.chain.from_iterable(map(tokenize, data_stream))
        write_output(token_stream, out_file, out_dir)

    if 'tokenize_dataset' in flags:
        data_stream = load_data(data_dir, input_file)
        output_file = "microblog2011_tokenized.txt"

        if show_progress:
            data_stream = tqdm.tqdm(data_stream, total = line_count, unit = "lines")

        tokenize_dataset(data_stream, output_file)

    if 'tokenize_dataset_first20' in flags:
        data_stream = load_data(data_dir, input_file)
        data_stream = itertools.islice(data_stream, 20)
        output_file = "microblog2011_tokenized_first20.txt"

        if show_progress:
            data_stream = tqdm.tqdm(data_stream, total = 20, unit = "lines")

        tokenize_dataset(data_stream, output_file)

    # tokens include \n EOL character
    def token_stream():
        data_stream = load_data(data_dir, input_file)

        if show_progress:
            data_stream = tqdm.tqdm(data_stream, total = line_count, unit = "lines")
        return itertools.chain.from_iterable(map(tokenize, data_stream))

    if 'token_counts' in flags:
        print("Calculating token counts.")
        token_counter = Counter(token_stream())

        """
        SECTION: B
        How many tokens did you find in the corpus? 
        How many types (unique tokens) did you have? 
        What is the type/token ratio for the corpus? 
        The type/token ratio is defined as the number of types divided by the number of tokens.
        """
        total_token_count = sum(token_counter.values())
        unique_token_count = len(token_counter)
        type_token_ratio = total_token_count / unique_token_count

        output = {
            "total_token_count": total_token_count,
            "unique_token_count": unique_token_count,
            "type_token_ratio": type_token_ratio
        }
        write_json(output, "b.json")

        """
        SECTION: C
        For each token, print the token and its frequency in a file called Tokens.txt 
        (from the most frequent to the least frequent) and include the first 100 lines in your report.
        """
        sorted_token_counter = token_counter.most_common()
        write_counter(sorted_token_counter, out_dir, "Tokens.txt")

        """
        SECTION: D 
        How many tokens appeared only once in the corpus?
        """
        once_only_tokens = (token for token in token_counter if token_counter[token] == 1)
        write_list(once_only_tokens, out_dir, "d.txt")        

        """
        SECTION: E
        From the list of tokens, extract only words, by excluding punctuation and other symbols. 
        How many words did you find? List the top 100 most frequent words in your report, with their frequencies. 
        What is the type/token ratio when you use only word tokens (called lexical diversity)?
        """
        word_regex = re.compile("^[a-z]+$", re.I)
        word_counter = Counter({token: token_counter[token] for token in token_counter if word_regex.match(token)})
        unique_words = len(word_counter)
        word_count = sum(word_counter.values())

        most_common_100_words = word_counter.most_common(100)
        output = {
            "unique_words": unique_words,
            "lexical_diversity": unique_words / word_count,
            "most_common_100_words": most_common_100_words,
        }
        write_json(output, "e.json")

        """
        SECTION: F
        From the list of words, exclude stopwords.
        List the top 100 most frequent words and their frequencies.
        You can use this list of stopwords (or any other that you consider adequate).
        """
        stop_words = set(load_data(data_dir, "StopWords.txt"))
        non_stop_words_counter = Counter({word: word_counter[word] for word in word_counter if word not in stop_words})
        most_common_non_stopwords = non_stop_words_counter.most_common()
        write_counter(most_common_non_stopwords, out_dir, "f.txt")

    if 'pairs' in flags:
        pass
        """
        SECTION: G
        Compute all the pairs of two consecutive words (excluding stopwords and punctuation).
        List the most frequent 100 pairs and their frequencies in your report. 
        Also compute the type/token ratio when you use only word tokens without stopwords (called lexical density)?
        """

    """
    SECTION: H
    Extract multi-word expressions (composed of two or more words, so that the meaning of the expression is more than the composition of the meanings of its words). 
    You can use an existing tool or your own method (explain what tool or method you used). 
    List the most frequent 100 expressions extracted. 
    Make sure they are multi-word expressions and not just n-grams or collocations.
    """

if __name__ == "__main__":
    """
    Make sure there are the following directory structure
    - code
    - data
    - \ microblog2011.txt
    - output

    Run the script from the <code> directory
    """

    run_all(
        show_progress = True,
        # tokenize_dataset = True,
        # tokenize_dataset_first20 = True,
        token_counts = True,

    )