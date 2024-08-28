import random
import datasets

SEPARATOR = '<<<SEP>>>'


#DATASETS = ['writing', 'english', 'german', 'pubmed']
DATASETS = ['Imdb', 'Tweet', 'news', 'writing_prompts', 'xsum']
def load_Imdb(datapath, cache_dir=None):
    imdb_path_cover = 'data/aclImdb/cover'
    imdb_path_stego = datapath
    with open(f'{imdb_path_cover}.txt', 'r', encoding='utf-8') as fc:
        reviews_c = fc.readlines()
    filtered_c = [process_Imdb(review) for review in reviews_c]
    with open(f'{imdb_path_stego}.txt', 'r', encoding='utf-8') as fs:
        reviews_s = fs.readlines()
    filtered_s = [process_Imdb(review) for review in reviews_s]
    random.seed(0)
    random.shuffle(filtered_c)
    random.shuffle(filtered_s)
    # with open(f'{imdb_path}1.txt', 'w', encoding='utf-8') as f:
    #     passages = [filter + '\n' for filter in filtered]
    #     f.writelines(passages)
    return filtered_c, filtered_s

def load_writing_prompts(datapath, cache_dir=None):
    imdb_path_cover = 'data/writing_prompts/writing_prompts_cover'
    imdb_path_stego = datapath
    with open(f'{imdb_path_cover}.txt', 'r', encoding='utf-8') as fc:
        reviews_c = fc.readlines()
    filtered_c = [process_Imdb(review) for review in reviews_c]
    with open(f'{imdb_path_stego}.txt', 'r', encoding='utf-8') as fs:
        reviews_s = fs.readlines()
    filtered_s = [process_Imdb(review) for review in reviews_s]
    random.seed(0)
    random.shuffle(filtered_c)
    random.shuffle(filtered_s)
    # with open(f'{imdb_path}1.txt', 'w', encoding='utf-8') as f:
    #     passages = [filter + '\n' for filter in filtered]
    #     f.writelines(passages)
    return filtered_c, filtered_s
def load_xsum(datapath, cache_dir=None):
    imdb_path_cover = 'data/xsum/xsum_cover'
    imdb_path_stego = datapath
    with open(f'{imdb_path_cover}.txt', 'r', encoding='utf-8') as fc:
        reviews_c = fc.readlines()
    filtered_c = [process_Imdb(review) for review in reviews_c]
    with open(f'{imdb_path_stego}.txt', 'r', encoding='utf-8') as fs:
        reviews_s = fs.readlines()
    filtered_s = [process_Imdb(review) for review in reviews_s]
    random.seed(0)
    random.shuffle(filtered_c)
    random.shuffle(filtered_s)
    # with open(f'{imdb_path}1.txt', 'w', encoding='utf-8') as f:
    #     passages = [filter + '\n' for filter in filtered]
    #     f.writelines(passages)
    return filtered_c, filtered_s
def load_news(cache_dir=None):
    imdb_path_cover = 'data/news/ac/cover'
    imdb_path_stego = 'data/news/ac/stego'
    with open(f'{imdb_path_cover}.txt', 'r', encoding='utf-8') as fc:
        news_c = fc.readlines()
    filtered_c = [process_news(review) for review in news_c]
    length = len(filtered_c)
    tmpc = []
    for i in range(length // 10):
        tmpc.append('. '.join(filtered_c[i : i+10]))
    processed_tmp = [process_Imdb(news) for news in tmpc]
    filtered_c = processed_tmp
    with open(f'{imdb_path_stego}.txt', 'r', encoding='utf-8') as fs:
        news_s = fs.readlines()
    filtered_s = [process_news(review) for review in news_s]
    length = len(filtered_s)
    tmps = []
    for i in range(length // 10):
        tmps.append('. '.join(filtered_s[i: i + 10]))
    processed_tmp = [process_Imdb(news) for news in tmps]
    filtered_s = processed_tmp
    random.seed(0)
    random.shuffle(filtered_c)
    random.shuffle(filtered_s)
    # with open(f'{imdb_path_cover}1.txt', 'w', encoding='utf-8') as f:
    #     passages = [filter + '\n' for filter in filtered_c]
    #     f.writelines(passages)
    return filtered_c, filtered_s

def process_Imdb(review):
    return review.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').replace(
        '<br /><br />', '\t').strip()
def process_news(news):
    return news.capitalize().replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').replace(
        '<br /><br />', '\t').strip()
def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()

#
# def load_writing(cache_dir=None):
#     writing_path = 'data/writingPrompts'
#
#     with open(f'{writing_path}/valid.wp_source', 'r') as f:
#         prompts = f.readlines()
#     with open(f'{writing_path}/valid.wp_target', 'r') as f:
#         stories = f.readlines()
#
#     prompts = [process_prompt(prompt) for prompt in prompts]
#     joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
#     filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]
#
#     random.seed(0)
#     random.shuffle(filtered)
#
#     return filtered


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')

