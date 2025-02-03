import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import ssl
import spacy
import pytextrank

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')

def summarize_text(text: str, sentence_count: int = 3) -> str:
    """Summarizes input text using TextRank algorithm."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    
    # Generate summary with the top N sentences
    summary = summarizer(parser.document, sentence_count)
    
    return " ".join(str(sentence) for sentence in summary)

# Example text
text = """Natural Language Processing (NLP) is a field of AI that enables computers to understand human language.
It involves techniques such as tokenization, stemming, and machine learning.
Text summarization is an important application of NLP, where long texts are reduced while preserving meaning.
One popular algorithm for extractive summarization is TextRank, inspired by Google's PageRank algorithm.
TextRank builds a graph of sentences and ranks them based on importance.
Higher-ranked sentences are used to generate a summary."""

# Generate a summary
summary = summarize_text(text, sentence_count=2)
print("\n===== Summary =====")
print(summary)

print('Original Document Size:', len(summary))
doc = nlp(summary)

for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=2):
    print(sent)
    print('Summary Length:', len(sent))
