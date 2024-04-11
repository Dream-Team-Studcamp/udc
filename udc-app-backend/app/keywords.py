import numpy as np
import pymorphy2
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

EMBEDDER = SentenceTransformer('mlsa-iai-msu-lab/sci-rus-tiny')

MORPH = pymorphy2.MorphAnalyzer()

# правило — последовательность слов (список) с их частями речи и падежами.
MORPH_RULES = [
    [{'pos': 'NOUN', 'case': 'nomn'}],
    [{'pos': 'NOUN', 'case': 'accs'}],
    [{'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'NOUN', 'case': 'loct'}],
    [{'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'nomn'}],
    [{'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'accs'}],
    [{'pos': 'ADJF', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'nomn'}],
    [{'pos': 'ADJF', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'accs'}],
    [{'pos': 'ADJF', 'case': 'gent'}, {'pos': 'NOUN', 'case': 'loct'}],
    [{'pos': 'ADJF', 'case': 'gent'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'NOUN', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'NOUN', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'NOUN', 'case': 'gent'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'NOUN', 'case': 'nomn'}, {'pos': 'ADJF', 'case': 'gent'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'NOUN', 'case': 'accs'}, {'pos': 'ADJF', 'case': 'gent'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'nomn'}],
    [{'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'accs'}],
    [{'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'ADJF', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'nomn'}],
    [{'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'ADJF', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'accs'}],
    [{'pos': 'ADJF', 'case': 'accs'}, {'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'nomn'}],
    [{'pos': 'ADJF', 'case': 'accs'}, {'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'accs'}],
    [{'pos': 'ADJF', 'case': 'accs'}, {'pos': 'ADJF', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'nomn'}],
    [{'pos': 'ADJF', 'case': 'accs'}, {'pos': 'ADJF', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'accs'}],
    [{'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'ADJF', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'ADJF', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'nomn'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'ADJF', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'accs'}, {'pos': 'NOUN', 'case': 'gent'}],
    [{'pos': 'ADJF', 'case': 'gent'}, {'pos': 'NOUN', 'case': 'gent'}, {'pos': 'NOUN', 'case': 'gent'}]
]


def convert_parsed_to_morph_rule_format(parsed_word):
    pos = parsed_word.tag.POS
    case_ = parsed_word.tag.case if pos in ['NOUN', 'ADJF'] else None
    return {'pos': pos, 'case': case_}


def represent_tokenized_text_as_morphs(tokenized_text, morph):
    res = []
    res_in_morph_rule_format = []
    for word in tokenized_text:
        parsed_word = morph.parse(word)[0]  # most probable one
        res.append(parsed_word)
        res_in_morph_rule_format.append(convert_parsed_to_morph_rule_format(parsed_word))
    return res, res_in_morph_rule_format


def match_morph_rule(tokens_as_morphs_in_morph_rule_format, morph_rule):
    return tokens_as_morphs_in_morph_rule_format == morph_rule


def get_raw_keywords_candidates(tokens_as_morphs, tokens_as_morphs_in_morph_rule_format, morph_rules_expanded,
                                deep=False):  # deep will extract terms recursively
    morph_rules_expanded = sorted(morph_rules_expanded, key=len, reverse=True)
    candidates = []
    i = 0
    while i < len(tokens_as_morphs) + 1:
        for rule in morph_rules_expanded:
            rule_len = len(rule)
            slice_ = tokens_as_morphs_in_morph_rule_format[i: i + rule_len]  # no overflow thanks to python
            if match_morph_rule(slice_, rule):
                candidates.append(tokens_as_morphs[i: i + rule_len])
                if not deep:
                    i += rule_len - 1
                    break
        i += 1
    return candidates


def get_raw_keywords(text, deep=False):
    global MORPH, MORPH_RULES  # global variables
    tokenized_text = word_tokenize(text)  # global-level function from nltk
    tokens_as_morphs, tokens_as_morphs_in_morph_rule_format = represent_tokenized_text_as_morphs(tokenized_text,
                                                                                                 morph=MORPH)
    raw_keywords = get_raw_keywords_candidates(tokens_as_morphs, tokens_as_morphs_in_morph_rule_format,
                                               morph_rules_expanded=MORPH_RULES, deep=deep)
    return raw_keywords


def handle_inflection_error(word, forms):
    try:
        return word.inflect(forms).word
    except (AttributeError, ValueError):
        return word.word


def normalize_keywords(raw_keywords):
    normalized_keywords = []

    for kws in raw_keywords:
        if str(kws[0].tag) == 'LATN':
            normalized_keywords.append(' '.join([kw.word for kw in kws]))

        elif len(kws) == 1:
            parsed_noun = kws[0]
            if parsed_noun.tag.number != 'plur':
                parsed_noun = parsed_noun.normal_form
            else:
                parsed_noun = handle_inflection_error(parsed_noun, {'nomn', 'plur'})
            normalized_keywords.append(parsed_noun)

        elif len(kws) == 2:
            if kws[0].tag.POS == 'NOUN' and kws[1].tag.POS == 'NOUN':
                normalized_keywords.append(kws[0].normal_form + ' ' + kws[1].word)
            elif kws[0].tag.POS == 'ADJF' and kws[1].tag.POS == 'NOUN':
                parsed_adj = kws[0]
                parsed_noun = kws[1]
                if parsed_noun.tag.number != 'plur':
                    inflected_adj = handle_inflection_error(parsed_adj, {parsed_noun.tag.gender, 'nomn', 'sing'})
                    parsed_noun = parsed_noun.normal_form
                else:
                    inflected_adj = handle_inflection_error(parsed_adj, {'nomn', 'plur'})
                    parsed_noun = handle_inflection_error(parsed_noun, {'nomn', 'plur'})
                normalized_keywords.append(inflected_adj + ' ' + parsed_noun)

        elif len(kws) == 3:
            if kws[0].tag.POS == 'NOUN':
                parsed_noun = kws[0]
                if parsed_noun.tag.number != 'plur':
                    parsed_noun = parsed_noun.normal_form
                else:
                    parsed_noun = handle_inflection_error(parsed_noun, {'nomn', 'plur'})
                normalized_keywords.append(parsed_noun + ' ' + kws[1].word + ' ' + kws[2].word)
            elif kws[0].tag.POS == 'ADJF' and kws[1].tag.POS == 'NOUN':
                parsed_adj = kws[0]
                parsed_noun = kws[1]
                if parsed_noun.tag.number != 'plur':
                    inflected_adj = handle_inflection_error(parsed_adj, {parsed_noun.tag.gender, 'nomn', 'sing'})
                    parsed_noun = parsed_noun.normal_form
                else:
                    inflected_adj = handle_inflection_error(parsed_adj, {'nomn', 'plur'})
                    parsed_noun = handle_inflection_error(parsed_noun, {'nomn', 'plur'})
                normalized_keywords.append(inflected_adj + ' ' + parsed_noun + ' ' + kws[2].word)
            elif kws[0].tag.POS == 'ADJF' and kws[1].tag.POS == 'ADJF' and kws[2].tag.POS == 'NOUN':
                parsed_adj0, parsed_adj1, parsed_noun = kws
                if parsed_noun.tag.number != 'plur':
                    inflected_adj0 = handle_inflection_error(parsed_adj0, {parsed_noun.tag.gender, 'nomn', 'sing'})
                    inflected_adj1 = handle_inflection_error(parsed_adj1, {parsed_noun.tag.gender, 'nomn', 'sing'})
                    parsed_noun = parsed_noun.normal_form
                else:
                    inflected_adj0 = handle_inflection_error(parsed_adj0, {'nomn', 'plur'})
                    inflected_adj1 = handle_inflection_error(parsed_adj1, {'nomn', 'plur'})
                    parsed_noun = handle_inflection_error(parsed_noun, {'nomn', 'plur'})
                normalized_keywords.append(inflected_adj0 + ' ' + inflected_adj1 + ' ' + parsed_noun)

    return normalized_keywords


def bow_keywords_selection(keywords_candidates, threshold=0.05):
    keywords_candidates_array = np.array(keywords_candidates)
    words, counts = np.unique(keywords_candidates_array, return_counts=True)
    frequencies = counts / len(words)

    sorted_indices = np.argsort(-counts)
    sorted_words = words[sorted_indices]
    sorted_frequencies = frequencies[sorted_indices]
    return [word for i, word in enumerate(sorted_words) if sorted_frequencies[i] > threshold]


def neural_keywords_selection(abstract, keywords_candidates, n=7):
    global EMBEDDER
    if not keywords_candidates or n <= 0:
        return []
    abstract_embedding = EMBEDDER.encode([abstract])
    kws_embeddings = EMBEDDER.encode(keywords_candidates)
    data = cosine_similarity(abstract_embedding, kws_embeddings)
    keywords = [keywords_candidates[i] for i in np.argsort(-data.flatten())[:n]]
    return keywords


def get_keywords(abstract, n=7, bow_threshold=0.05):
    raw_keywords = normalize_keywords(get_raw_keywords(abstract, deep=False))
    raw_keywords_deep = normalize_keywords(get_raw_keywords(abstract, deep=True))

    keywords_bow = list(set(bow_keywords_selection(raw_keywords_deep, bow_threshold)))
    raw_keywords = list(set(raw_keywords) - set(keywords_bow))

    keywords_bow_neural = neural_keywords_selection(abstract, raw_keywords, n=(n - len(keywords_bow)))
    return (keywords_bow + keywords_bow_neural)[:n]
