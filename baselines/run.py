import os

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--dataset", help="dataset name", type=str, default="wos")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--context", type=str, default='none')
parser.add_argument("--no_test", action="store_true")
parser.add_argument("--max_desc_token", type=int, default=32)
parser.add_argument("--pre_classify", action="store_true")
parser.add_argument("--sentence_filter", action="store_true")
parser.add_argument("--graphrag", action="store_true")
parser.add_argument("--compress_desc", action="store_true")
parser.add_argument("--compress", type=str, default='none')
parser.add_argument("--compress_rate", type=float, default=0.8)
parser.add_argument("--LLM", type=str, default='llama3')
parser.add_argument("--keyword_only", action="store_true")
parser.add_argument("--desc_keywords", action="store_true")
parser.add_argument("--index_label_only", action="store_true")
parser.add_argument("--steiner_tree", action="store_true")
parser.add_argument("--shot", type=int, default=0)
parser.add_argument("--edge_weighting", type=str, default="unit")
parser.add_argument("--no_label_name", action="store_true")
parser.add_argument("--round", type=int, default=4)
parser.add_argument("--online_index", type=str, default='none',
                    help='all or filtered. Index when response in all labels or filtered labels')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
print(args)
# print(args.gpu)
# print(os.environ['CUDA_VISIBLE_DEVICES'])
from huggingface_hub import login

login("YOUR TOKEN HERE")
from utils import *

check_device()
from dataset import *

from torch.utils.data import DataLoader, ConcatDataset
import json
import time
from GORAG import GORAG
import tqdm
from itertools import combinations, chain
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

print('Starting Time:')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
setup_seed(args.seed)
device = torch.device(f"cuda:{args.gpu}")


def desc_label(GORAG, candidate_label: list, short_desc=False):
    label_desc = {}
    if context == 'KG':
        for label_name in candidate_label:
            desc = wikidata_search(label_name)
            if desc:
                label_desc[label_name] = f"{label_name}: {desc}"
            else:
                no_desc.append(f"{label_name}: No description found")
    elif context == 'LLM':
        for label_name in candidate_label:
            if args.graphrag and not short_desc:
                desc = GORAG.LLM_search(label_name, -1)
            else:
                desc = GORAG.LLM_search(label_name, args.max_desc_token)
            if args.compress == 'text' and args.compress_desc:
                desc = llm_lingua.compress_prompt(
                    [desc],
                    rate=args.compress_rate,
                    force_tokens=['\n', '?'],
                    use_sentence_level_filter=use_sentence_level_filter, )[
                    'compressed_prompt']
            label_desc[label_name] = f"{label_name}: {desc}"
    if label_desc:
        total_label_desc.extend(label_desc)
    return label_desc


def test(test_loader: DataLoader,
         gorag_model: GORAG,
         round_num: str,
         candidate_label=None,
         # label_desc=None,
         # max_desc_label: int = 3,
         label_desc_short=None,
         online_index_round=False
         ):
    if label_desc_short is None:
        label_desc_short = {}
    if args.online_index != 'none' and online_index_round:
        print(f'Online Indexing at round {round_num}! ')
    print('==================')
    ans_list = []
    total_res = []
    truth_label = []
    label_list = []
    bad_case = []
    good_case = []
    NO_KG_MATCHING = 0
    NO_PATH = 0
    FUZZY = 0
    FIND_CATEGORY = 0
    for item in tqdm.tqdm(test_loader, desc=f'{round_num} Testing'):
        prompt = item['doc_token'][0]
        label = item[f'label'][0]
        # label_num = item[f'label_#'][0]
        un_contained_keyword = set()
        KG_selected_label = set()
        if args.online_index != 'none' and online_index_round:
            clean_corpus_list_oi = re.findall(r'(?u)\b\w\w+\b', prompt)
            cleaned_corpus.append(str(' '.join(clean_corpus_list_oi)))
        if args.compress == 'text':
            # prompt = llm_lingua.compress_prompt(prompt, instruction="", question="", target_token=100)
            prompt = llm_lingua.compress_prompt(
                [prompt],
                rate=args.compress_rate,
                force_tokens=['\n', '?'],
                use_sentence_level_filter=use_sentence_level_filter, )[
                'compressed_prompt']
        if args.graphrag:
            keyword_list, keyword_entropy = GORAG.LLM_Extraction(prompt, -1)
            keyword_keyword_list = []
            if args.hie_keywords_test:
                for keyword_keyword in keyword_list:
                    keyword_extract_prompt = [{"role": "user", "content": f'What is {keyword_keyword}?'}]
                    keyword_desc, _ = GORAG.chat(keyword_extract_prompt, -1)
                    keyword_keyword_response, keyword_keyword_entropy = GORAG.LLM_Extraction(keyword_desc, -1,
                                                                                             target='keyword')
                    keyword_keyword_list.extend(keyword_keyword_response)
            keyword_set = set()

            for keyword in keyword_list + keyword_keyword_list:
                keyword = process_keyword(keyword)
                if keyword.lower() in GORAG.Graph.nodes():
                    keyword_set.add(keyword.lower())
                else:
                    un_contained_keyword.add(keyword.lower())
            if keyword_set:
                if args.steiner_tree:
                    paths = []
                    nei_triple_list = []
                    try:
                        st_nodes = GORAG.get_steiner_tree_nodes(list(keyword_set))
                    except KeyError:
                        for wrong_keyword in keyword_set:
                            if wrong_keyword.lower() not in GORAG.Graph.nodes():
                                print(wrong_keyword)
                        raise KeyError
                    for node in st_nodes:
                        if node in candidate_label:
                            KG_selected_label.add(node)
                else:
                    count_dict, paths = GORAG.find_path(list(keyword_set), 'research area')
                    nei_triple_list = GORAG.return_neighborhood(list(keyword_set))
                    if count_dict:
                        # Sort the count_dict w.r.t to its value
                        sorted_count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
                        for node, count in sorted_count_dict:
                            if node in candidate_label:
                                KG_selected_label.add(node)
            else:
                paths, nei_triple_list = [], []
            if not args.steiner_tree and nei_triple_list or paths:
                if not paths:
                    NO_PATH += 1

                for triples in nei_triple_list:
                    components = triples.lower().split('[SEP]'.lower())
                    if components[0] in candidate_label:
                        KG_selected_label.add(components[0])
                    if components[1] in candidate_label:
                        KG_selected_label.add(components[1])

            if KG_selected_label:
                label_desc_filtered = []
                KG_selected_label = list(KG_selected_label)
                for category in KG_selected_label:
                    if not args.no_label_name:
                        label_desc_filtered.append(label_desc_short[category])
                    else:
                        label_desc_filtered.append(', '.join(label_desc_short[category]))
                FIND_CATEGORY += 1
                desc_text = '\n'.join(label_desc_filtered)
                if args.keyword_only and keyword_list:
                    content = (
                        f"Now you need to classify texts into one of the following classes: \n{', '.join(KG_selected_label)}.\n"
                        f"Here are some descriptions for each of the above mentioned classes stating their differences:\n{desc_text}.\n "
                        # f"Here is the candidate label in KG based on the keywords extracted from the text:\n{', '.join(KG_selected_label)}.\n"
                        f"Please answer strictly by only output one of the above mentioned class based on the following keywords of the text, "
                        f"do not output any other words: \n{', '.join(keyword_list)}")
                else:
                    content = (
                        f"Now you need to classify texts into one of the following classes: \n{', '.join(KG_selected_label)}.\n"
                        f"Here are some descriptions for each of the above mentioned classes stating their differences:\n{desc_text}.\n "
                        # f"Here is the candidate label in KG based on the keywords extracted from the text:\n{', '.join(KG_selected_label)}.\n"
                        f"Please answer strictly by only output one of the above mentioned class based on the following text, do not output any other words: \n{prompt}")
            else:
                label_desc_filtered = []
                for category in candidate_label:
                    if not args.no_label_name:
                        label_desc_filtered.append(label_desc_short[category])
                    else:
                        label_desc_filtered.append(', '.join(label_desc_short[category]))
                desc_text = '\n'.join(label_desc_filtered)
                if args.keyword_only and keyword_list:
                    if not args.no_label_name:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(candidate_label)}.\n"
                            f"Here are some descriptions for each of the above mentioned classes stating their differences:\n{desc_text}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following keywords of the text,"
                            f" do not output any other words: \n{', '.join(keyword_list)}")
                    else:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(candidate_label)}.\n"
                            f"Here are some keywords for each of the above mentioned classes representing their features:\n{desc_text}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following keywords of the text,"
                            f" do not output any other words: \n{', '.join(keyword_list)}")
                else:
                    if not args.no_label_name:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(candidate_label)}.\n"
                            f"Here are some descriptions for each of the above mentioned classes stating their differences:\n{desc_text}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following text, do not output any other words: \n{prompt}")
                    else:
                        content = (
                            f"Now you need to classify texts into one of the following classes: \n{', '.join(candidate_label)}.\n"
                            f"Here are some keywords for each of the above mentioned classes representing their features:\n{desc_text}.\n"
                            f"Please answer strictly by only output one of the above mentioned class based on the following text, do not output any other words: \n{prompt}")
        else:
            if args.context != 'none':
                label_desc_list = [' ,'.join(keyword_l) for keyword_l in label_desc_short.values()]
                label_desc_str = '; '.join(label_desc_list)
                content = (
                    f"Now you need to classify texts into one of the following classes: {', '.join(candidate_label)}. "
                    f"Here are some keywords for each of the above mentioned classes stating their differences: {label_desc_str}. "
                    f"Please answer shortly by only give out one of the above mentioned class based on the following text: '{prompt}'")
            else:
                content = (
                    f"Now you need to classify texts into one of the following classes: {', '.join(candidate_label)}. "
                    # f"Here are some descriptions for each of the above mentioned classes stating their differences: {'; '.join(label_desc)}. "
                    f"Please answer shortly by only give out one of the above mentioned class based on the following text: '{prompt}'")
        if args.compress == 'prompt':
            question = f"What is the best suited class describing this text? "
            instruction = (
                f"Now you need to classify texts into one of the following classes: "
                f"{', '.join(KG_selected_label if KG_selected_label else candidate_label)}. "
                f"Answer shortly by only give out one of the mentioned classes, not any other words. "
                f"Here are some descriptions for the classes mentioned above: \n{label_desc_short}. "
            )
            content = llm_lingua.compress_prompt(
                [prompt],
                question=question,
                instruction=instruction,
                rate=args.compress_rate,
                # Set the special parameter for LongLLMLingua
                condition_in_question="after_condition",
                reorder_context="sort",
                dynamic_context_compression_ratio=0.3,  # or 0.4
                condition_compare=True,
                context_budget="+100",
                rank_method="longllmlingua",
                use_sentence_level_filter=use_sentence_level_filter,
                keep_first_sentence=1,
            )['compressed_prompt']

        messages_test = [{"role": "user", "content": content}]
        res, res_score = GORAG.chat(messages_test, max_new_token)
        cleaned_res = re.findall(r'(?u)\b\w\w+\b', res)
        cleaned_res = ' '.join(cleaned_res)
        cleaned_res.replace('-', ' ')
        cleaned_res.replace("'s", ' ')
        cleaned_res.replace("_", ' ')
        if cleaned_res and args.graphrag and un_contained_keyword and cleaned_res.lower().strip() in candidate_label:
            if args.online_index == 'all' and online_index_round:
                GORAG.online_indexing(un_contained_keyword,
                                      cleaned_res,
                                      args,
                                      candidate_label,
                                      cleaned_corpus,
                                      inferenced_corpus)
            elif args.online_index == 'filtered' and online_index_round:
                GORAG.online_indexing(un_contained_keyword,
                                      cleaned_res,
                                      args,
                                      KG_selected_label,
                                      cleaned_corpus,
                                      inferenced_corpus)
        if args.predict_confidence:
            for label_name in candidate_label:
                token_probs = []
                if label_name.lower() in cleaned_res.lower():
                    for i, score in enumerate(res_score[1]):
                        # Convert the scores to probabilities
                        probs = torch.softmax(score, -1)
                        # Take the probability for the generated tokens (at position i in sequence)
                        token_probs.append(probs[0, res_score[0][i]].item())
                    prod = 1
                    for x in token_probs:
                        prod *= x
                    inferenced_corpus.append((prompt.lower(), label_name.lower(), prod))
                    break
        if label.lower().strip() == cleaned_res.lower().strip():
            ans_list.append(label2id[cleaned_res.lower().strip()])
            good_case_dict = {
                'text': prompt,
                'prompt': content,
                'label': item['label'][0],
                # 'label_#': truth_label_num,
                'response': cleaned_res,
            }
            good_case.append(good_case_dict)
        else:
            if cleaned_res.lower().strip() in label2id:
                ans_list.append(label2id[cleaned_res.lower().strip()])
            else:
                ans_list.append(len(label2id))
            bad_case_dict = {
                'text': prompt,
                'prompt': content,
                'label': item['label'][0],
                # 'label_#': truth_label_num,
                'response': cleaned_res,
            }
            bad_case.append(bad_case_dict)
        # label_list.append(label)
        label_list.append(label2id[label.lower().strip()])
    print(f'No KG matching: {NO_KG_MATCHING}')
    print(f'No path: {NO_PATH}')
    print(f'Fuzzy matched: {FUZZY}')
    print(f'Find category: {FIND_CATEGORY}')
    return ans_list, total_res, truth_label, bad_case, good_case, label_list


setup_seed(args.seed)
context = args.context
use_sentence_level_filter = args.sentence_filter
total_label_desc = []
no_desc = []
inferenced_corpus = []
max_new_token = 32
labels = []
if args.dataset == 'wos':
    with open(f'../dataset/id2label.json', 'r') as file:
        id2label = json.load(file)
    with open(f'../dataset/label2id.json', 'r') as file:
        label2id = json.load(file)
    for key, value in id2label.items():
        # labels.append(f'{int(key)}: {value}')
        cleaned_label = f'{value}'.lower()
        cleaned_label = re.findall(r'(?u)\b\w\w+\b', cleaned_label)
        cleaned_label = ' '.join(cleaned_label)
        labels.append(str(cleaned_label))
    total_label_list = list(label2id.keys())
# labels = list(id2label.values())
else:
    for i in range(31):
        labels.append(f'label {i}')
if args.compress == 'prompt':
    llm_lingua = PromptCompressor(model=GORAG.LLM_model, tokenizer=GORAG.LLM_tokenizer)
elif args.compress == 'text':
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True
    )
GORAG = GORAG(args)

round_father_label_dict = {
    'n1': ['CS', 'ECE'],
    'n2': ['Medical'],
    'n3': ['Civil', 'Psychology'],
    'n4': ['MAE', 'biochemistry'],
}

round_data_dict = {
    'n1': {},
    'n2': {},
    'n3': {},
    'n4': {},
}
if args.round == 6:
    round_data_dict['n5'] = {}
    round_data_dict['n6'] = {}
elif args.round == 8:
    round_data_dict['n5'] = {}
    round_data_dict['n6'] = {}
    round_data_dict['n7'] = {}
    round_data_dict['n8'] = {}

if args.dataset == 'wos':
    if args.round == 6:
        round_label_dict = load_wos_taxnomy(6)
    elif args.round == 8:
        round_label_dict = load_wos_taxnomy(8)
    else:
        round_label_dict = load_wos_taxnomy(4)
else:
    round_label_dict = {
        'n1': [f'label {i}' for i in range(8)],
        'n2': [f'label {i}' for i in range(8, 16)],
        'n3': [f'label {i}' for i in range(16, 24)],
        'n4': [f'label {i}' for i in range(24, 31)],
    }
exp_name = return_exp_name(args)

for round_num, data_dict in round_data_dict.items():
    for split in ['train', 'test']:
        if args.dataset == 'wos':
            round_dataset = WOS4RoundDataset(
                f'../dataset/wos_{args.shot}_shot_{args.round}/split/{round_num}/{split}.txt',
                label2id)
        else:
            round_dataset = ReutersDataset(
                f'../dataset/reuters_{args.shot}_shot/split/{round_num}/{split}.txt')
        # print(f'{round_num} {split} dataset loaded!')
        # print(len(round_dataset))
        data_dict[split] = DataLoader(round_dataset, batch_size=1, shuffle=False)
# print(len(round_data_dict))
if args.graphrag:
    batch_size = 1
else:
    batch_size = 64

sample_text = ('this paper presents a novel method for the analysis of nonlinear financial and economic systems. '
               'the modeling approach integrates the classical concepts of state space representation and time series regression. '
               'the analytical and numerical scheme leads to a parameter space representation that constitutes a valid alternative to represent the dynamical behavior. '
               'the results reveal that business cycles can be clearly revealed, while the noise effects common in financial indices can '
               'elegantly be filtered out of the results.')

# print(sample_text)
current_labels = []
cleaned_corpus = []
label_desc_short = {}
current_data = []
for round_num, data_dict in round_data_dict.items():
    current_labels.append(round_label_dict[round_num])
    if context != 'none' and not args.no_label_name:
        label_desc = desc_label(GORAG, round_label_dict[round_num], short_desc=True)
        # print(label_desc)
        if not label_desc:
            # print(round_label_dict[round_num])
            raise ValueError('No label description found!')
        messages = [
            {"role": "system", "content": "You are an agent classifying texts into different classes. "
                                          f"Depending on their contents, you need to classify them strictly "
                                          f"into one of the following classes, separated by , : {labels}. "
                                          f"To help you better understand these classes, here are some descriptions "
                                          f"of each class: {'; '.join(list(label_desc.values()))}. "
             }
        ]
    else:
        label_desc = []
        messages = [
            {"role": "system", "content": "You are an agent classifying texts into different classes. "
                                          f"Depending on their contents, you need to classify them strictly "
                                          f"into one of the following classes, separated by , : {labels}. "
             }
        ]
    bad_case = []
    good_case = []
    count = 0
    NO_KG_MATCHING = 0
    current_round_candidate_labels = list(chain(*current_labels))
    if args.graphrag:
        if args.no_label_name:
            label_keywords_current_round = GORAG.generate_keywords(data_dict['train'])
            label_desc_short.update(label_keywords_current_round)
        else:
            # label_desc_short_current_round = desc_label(GORAG, round_label_dict[round_num], True)
            label_desc_short.update(label_desc)
        corpus_length_before_idx = len(cleaned_corpus)
        if not os.path.exists(
                f'./graph/{args.dataset}_{args.LLM}_{args.edge_weighting}_{round_num}_{args.round}.json') or args.online_index != 'none':
            index_start_time = time.time()
            if not args.no_label_name:
                for raw_desc in label_desc.values():
                    if isinstance(raw_desc, str):
                        clean_corpus_list = re.findall(r'(?u)\b\w\w+\b', raw_desc)
                        cleaned_corpus.append(str(' '.join(clean_corpus_list)))
            if not args.index_label_only:
                training_text_loader = data_dict['train']
                GORAG.index_training_text(training_text_loader, cleaned_corpus, args)
                training_text_length = len(training_text_loader)
            else:
                training_text_length = 0
            if not args.no_label_name:
                if args.index_label_only:
                    GORAG.get_tfidf_vectorizer(cleaned_corpus)
                for label_desc_idx, label_desc_key in enumerate(
                        tqdm.tqdm(label_desc.keys(), desc=f'Graph Index {round_num}')):
                    label_desc_value = label_desc[label_desc_key]
                    label = label_desc_value.split(':')[0]
                    desc = ':'.join(label_desc_value.split(':')[1:])
                    _ = GORAG.LLM_Construct(label=label,
                                            label_desc=desc,
                                            args=args,
                                            corpus=cleaned_corpus,
                                            text_idx=corpus_length_before_idx + training_text_length + label_desc_idx,
                                            candidate_labels=current_round_candidate_labels)
            GORAG.print_graph()
            index_end_time = time.time()
            print(f'Indexing time cost for {round_num} is: {(index_end_time - index_start_time) / 60} mins')
        else:
            with open(f'./graph/{args.dataset}_{args.LLM}_{args.edge_weighting}_{round_num}.json', 'r') as file:
                GORAG.load_node_link_data(json.load(file))

    # for message in messages:
    if not args.no_label_name:
        response, test_res_score = GORAG.chat(messages, max_new_token)
    # Done before here
    current_data.append(data_dict['test'])
    start_time = time.time()
    online_index = True
    if not args.no_test:
        for round_idx, current_round_test_data in enumerate(current_data[::-1]):
            # if round_idx == 0 and round_num == 'n1' and args.shot == 1:
            #     continue
            current_round_name = f'n{len(current_data) - round_idx}'
            # if current_round_name == round_num or round_num == 'n1':
            #     online_idx = True
            # else:
            #     online_idx = False
            ans_list, total_res, truth_label, bad_case, good_case, label_list = test(
                test_loader=current_round_test_data,
                gorag_model=GORAG,
                round_num=round_num,
                candidate_label=current_round_candidate_labels,
                label_desc_short=label_desc_short,  # Short label_desc or current round
                online_index_round=online_index
            )
            online_index = False
            end_time = time.time()
            acc = accuracy_score(y_true=label_list, y_pred=ans_list)
            prec = precision_score(y_true=label_list, y_pred=ans_list, average='weighted', zero_division=0)
            recall = recall_score(y_true=label_list, y_pred=ans_list, average='weighted', zero_division=0)
            f1 = f1_score(y_true=label_list, y_pred=ans_list, average='weighted', zero_division=0)
            print(f'The accuracy for {round_num}, {current_round_name} is: ', acc)
            print(f'The precision for {round_num}, {current_round_name} is: ', prec)
            print(f'The recall for {round_num}, {current_round_name} is: ', recall)
            print(f'The F1 for {round_num}, {current_round_name} is: ', f1)
            print(
                f'The inference time cost for {round_num}, {current_round_name} is: {(end_time - start_time) / 60} mins')
            if not os.path.exists(f'./llama_badcase_{round_num}_{args.round}'):
                os.makedirs(f'llama_badcase_{round_num}_{args.round}')
            try:
                with open(f'./llama_badcase_{round_num}_{args.round}/bad_case_{exp_name}.json', mode='w') as file:
                    file.write(json.dumps(bad_case, indent=4))
                with open(f'./llama_badcase_{round_num}_{args.round}/good_case_{exp_name}.json', mode='w') as file:
                    file.write(json.dumps(good_case, indent=4))
            except:
                continue
GORAG.print_graph()
print(exp_name)
