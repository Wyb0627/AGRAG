from torch.utils.data import DataLoader

from utils import *


class GORAG:
    def __init__(self, args):
        if args.LLM == 'llama3':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            # model_id = 'meta-llama/Meta-Llama-3-8B'
        elif args.LLM == 'mistral':
            model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
        elif args.LLM == 'qwen2.5':
            model_id = 'Qwen/Qwen2.5-7B-Instruct'
        elif args.LLM == 'qwen2':
            model_id = 'Qwen/Qwen2-7B-Instruct'
        elif args.LLM == 'llama3.1':
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        else:
            raise NotImplementedError('LLM not supported!')
        self.LLM_name = args.LLM
        self.Graph = nx.Graph()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.LLM_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if args.LLM == 'qwen':
            self.tokenizer.bos_token = '<|endoftext|>'

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.system_message = [{"role": "system",
                                "content": "You need to assign the given keyword to one of the most related topic provided. "
                                           "Respond only with the provided topic name, don't respond any extra words. "
                                }]

    def get_steiner_tree_nodes(self, terminal_node: list):
        steiner_graph = nx.algorithms.approximation.steiner_tree(self.Graph, terminal_node, weight='weight')
        return list(steiner_graph.nodes())

    def return_neighborhood(self, node_list: list[str], tuple_delimiter='[SEP]'):
        triples = []
        loaded_edges = []
        for node in node_list:
            if node in self.Graph.nodes():
                for neighbor in self.Graph.neighbors(node):
                    if f'{node} {neighbor}' in loaded_edges or f'{neighbor} {node}' in loaded_edges:
                        continue
                    loaded_edges.append(f'{node} {neighbor}')
                    relation = self.Graph[node][neighbor]['relation']
                    triples.append(f'({node}{tuple_delimiter}{neighbor}{tuple_delimiter}{relation})')
        return triples

    def node_link_data(self):
        print(f'Current graph:\n Nodes: {self.Graph.number_of_nodes()}, Edges: {self.Graph.number_of_edges()}')
        return json_graph.node_link_data(self.Graph)

    def print_graph(self):
        print(f'Current graph:\n Nodes: {self.Graph.number_of_nodes()}, Edges: {self.Graph.number_of_edges()}')

    def LLM_search(self, text: str, max_new_token: int, language='en'):
        if language == 'en':
            if max_new_token > 0:
                content = (
                    f"Describe the following academic research area shortly with less than {max_new_token} tokens: {text}. ")
            else:
                content = (
                    f"Describe the following academic research area: {text} ")
        else:
            if max_new_token > 0:
                content = (
                    f"简短描述以下用于Shell脚本分类的类别，使用不超过{max_new_token}字，该类别的不同层次以‘-’分隔：{text}。")
            else:
                content = (
                    f"描述以下用于Shell脚本分类的类别，该类别的不同层次以‘-’分隔：{text}。")
        prompt = [{"role": "user", "content": content}]
        response, res_score = self.chat(prompt, max_new_token)
        # content_label_prompt = ("Give me some keywords related to the following academic research area, "
        #                         f"split keywords with [SEP], please only give me the keywords: {text}")
        # prompt = [{"role": "user", "content": content_label_prompt}]
        # response_keywords = chat(prompt, max_new_token, model, tokenizer, terminators)
        # extracted_keywords = response_keywords.lower().split('[SEP]')
        return response.strip()

    def keyword_entropy(self, keyword: str, candidate_label, args):
        candidate_label_str = ', '.join(candidate_label)
        entropy_dict = {}
        for response in range(args.keyword_inference_num):
            content = f"Among the following provided topics: {candidate_label_str}, the keyword '{keyword}', are most related to topic: "
            message = self.system_message.copy()
            message.append({"role": "user", "content": content})
            response_str, response_tup = self.chat(message, 16)
            response_seq, response_score = response_tup[0], response_tup[1]
            # label_token = self.tokenizer([response_str], return_tensors="pt", add_special_tokens=False).input_ids
            try:
                entropy = semantic_entropy_score_only(response_seq, response_score)
                if entropy < 0:
                    continue
            except:
                # print(entropy)
                print(response_str)
                print(response_seq)
                raise RuntimeError
            if response not in entropy_dict:
                entropy_dict[response_str] = [entropy]
            else:
                entropy_dict[response_str].append(entropy)
        return entropy_dict

    def find_path(self, keyword_set: list[str], most_diverse_label: str, path_delimiter=' -> '):
        count_dict = {}
        paths = []
        for keyword_s, keyword_t in combinations(keyword_set, 2):
            dist_d2s = 999999999999999999999
            dist_d2t = 999999999999999999999
            # if args.cluster and args.diversity:
            if nx.has_path(self.Graph, most_diverse_label, keyword_s):
                dist_d2s = nx.shortest_path_length(self.Graph, most_diverse_label, keyword_s)
            if nx.has_path(self.Graph, most_diverse_label, keyword_t):
                dist_d2t = nx.shortest_path_length(self.Graph, most_diverse_label, keyword_t)
            if nx.has_path(self.Graph, keyword_s, keyword_t):
                shortest_node_path = nx.shortest_path(self.Graph, keyword_s, keyword_t)
                shortest_path = [keyword_s]
                for i in range(1, len(shortest_node_path)):
                    n1 = shortest_node_path[i - 1]
                    n2 = shortest_node_path[i]

                    # relation = Graph[n1][n2]['relation']
                    # shortest_path.extend([relation, n2])
                    shortest_path.append(n2)
                cut_down_dist_s = int(dist_d2s / 2) if most_diverse_label.lower() != 'research area' else dist_d2s - 1
                cut_down_dist_t = int(dist_d2t / 2) if most_diverse_label.lower() != 'research area' else dist_d2t - 1
                if len(shortest_path) >= cut_down_dist_s + cut_down_dist_t:
                    continue
                graph_shortest_path = path_delimiter.join(shortest_path)
                paths.append('(' + graph_shortest_path + ')')
                for node in shortest_path[:cut_down_dist_s] + shortest_path[-cut_down_dist_t:]:
                    if node in count_dict:
                        count_dict[node] += 1
                    else:
                        count_dict[node] = 1
        return count_dict, paths


    def remove_self_loop(self):
        self.Graph.remove_edges_from(nx.selfloop_edges(self.Graph))

    def index_training_text(self, training_loader: DataLoader, corpus, args):
        init_corpus_length = len(corpus)
        for data in training_loader:
            text = data['doc_token'][0]
            clean_text_list = re.findall(r'(?u)\b\w\w+\b', text)
            corpus.append(str(' '.join(clean_text_list)))
        if args.edge_weighting == 'tfidf':
            self.get_tfidf_vectorizer(corpus)
        for text_idx, data in enumerate(training_loader):
            text = data['doc_token'][0]
            data_label = data['label'][0]
            if args.no_label_name:
                text_keywords, text_keywords_entropy = self.LLM_Extraction(text, -1)
            else:
                text_keywords, text_keywords_entropy = self.LLM_Extraction(text, -1, label=data_label)
            for keyword_ori in text_keywords:
                keyword_ori = re.sub(r'related keyword\S', '', keyword_ori, flags=re.IGNORECASE)
                keyword = process_keyword(keyword_ori).lower()
                if not keyword or re.match(r'\W', keyword):
                    continue
                cleaned_keyword = re.findall(r'(?u)\b\w\w+\b', keyword)
                cleaned_keyword = ' '.join(cleaned_keyword)
                if args.edge_weighting == 'tfidf':
                    if cleaned_keyword not in self.tfidf_vectorizer.vocabulary_:
                        continue
                    tfidf_score = self.tfidf_predict(self.tfidf_vectorizer, self.tfidf_matrix, cleaned_keyword,
                                                     init_corpus_length + text_idx)
                    self.add_to_graph(keyword.lower(), data_label.lower(), 'related to', 1 - tfidf_score)
                    self.add_to_graph(data_label.lower(), 'label_connector', 'related to', 0.5)
                else:
                    self.add_to_graph(keyword.lower(), data_label.lower(), 'related to')
                    self.add_to_graph(data_label.lower(), 'label_connector', 'related to')

    def add_to_graph(self, n, v, r, w=1.0):
        if n not in self.Graph.nodes():
            self.Graph.add_node(n)
        if v not in self.Graph.nodes():
            self.Graph.add_node(v)
        if (n, v) not in self.Graph.edges():
            self.Graph.add_edge(n, v, relation=r, weight=w, count=1, total_weight=w)
        else:
            self.Graph[n][v]['total_weight'] += w
            self.Graph[n][v]['count'] += 1
            self.Graph[n][v]['weight'] = self.Graph[n][v]['total_weight'] / self.Graph[n][v]['count']
            # G[n][v]['total_weight'].append(w)
            # G[n][v]['count'] += 1
            # G[n][v]['weight'] = min(G[n][v]['total_weight'])

    def load_node_link_data(self, data):
        self.Graph = json_graph.node_link_graph(data)

    def LLM_Construct(self,
                      label: str,
                      label_desc: str,
                      args,
                      corpus=None,
                      text_idx=0,
                      candidate_labels=None):
        desc_keywords, desc_keywords_entropy = self.LLM_Extraction(label_desc, -1,
                                                                   target='desc')
        label_keywords_content = ('Give me some keywords related to the '
                                  f"academic research area: {label}, split keywords with [SEP]. "
                                  f"Please only reply me with related keywords, "
                                  f"don't reply anything else.")
        prompt = [{"role": "user", "content": label_keywords_content}]
        response, res_score = self.chat(prompt, -1)
        # print(response)
        label_keywords = response.split('[SEP]')
        keywords_relation = 'related to'
        skip = 0
        skip_old = 0
        graph_weight_list = []
        prediction_confidence = 1
        if args.edge_weighting != 'unit':
            self.add_to_graph(label.lower(), 'label_connector', keywords_relation, 0.5)
        else:
            self.add_to_graph(label.lower(), 'label_connector', keywords_relation)
        # keyword_set = set(desc_keywords + label_keywords + text_keywords)
        # if not args.desc_keywords:
        for keyword_ori in label_keywords:
            keyword_ori = re.sub(r'related keyword\S', '', keyword_ori, flags=re.IGNORECASE)
            keyword = process_keyword(keyword_ori).lower()
            if not keyword or re.match(r'\W', keyword):
                skip_old += 1
                continue
            cleaned_keyword = re.findall(r'(?u)\b\w\w+\b', keyword)
            cleaned_keyword = ' '.join(cleaned_keyword)
            if args.edge_weighting == 'tfidf' and cleaned_keyword not in self.tfidf_vectorizer.vocabulary_:
                continue
            if args.edge_weighting == 'semantic_entropy':
                keyword_token = self.tokenizer([keyword_ori], return_tensors="pt",
                                               add_special_tokens=False).input_ids
                # EntropyCal.keyword_entropy(keyword, label.lower(), candidate_label)
                keyword_entropy = semantic_entropy(keyword_token[0], res_score[0], res_score[1])
                self.add_to_graph(keyword.lower(), label.lower(), keywords_relation, keyword_entropy)
                graph_weight_list.append(keyword_entropy)
            elif args.edge_weighting == 'tfidf':
                try:
                    tfidf_score = self.tfidf_predict(self.tfidf_vectorizer, self.tfidf_matrix, cleaned_keyword,
                                                     text_idx)
                except IndexError:
                    print(self.tfidf_matrix.shape)
                    print(text_idx)
                    raise IndexError
                if 1 - tfidf_score <= 0:
                    print(cleaned_keyword)
                    print(tfidf_score)
                    raise ValueError
                self.add_to_graph(keyword.lower(), label.lower(), keywords_relation, 1 - tfidf_score)
                graph_weight_list.append(1 - tfidf_score)
            else:
                self.add_to_graph(keyword.lower(), label.lower(), keywords_relation)
        if args.desc_keywords:
            for keyword_ori, keyword_entropy in zip(desc_keywords,
                                                    desc_keywords_entropy):
                keyword_ori = re.sub(r'related keyword\S', '', keyword_ori, flags=re.IGNORECASE)
                keyword = process_keyword(keyword_ori).lower()
                if not keyword or re.match(r'\W', keyword):
                    skip_old += 1
                    continue
                cleaned_keyword = re.findall(r'(?u)\b\w\w+\b', keyword)
                cleaned_keyword = ' '.join(cleaned_keyword)
                if args.edge_weighting == 'tfidf' and cleaned_keyword not in self.tfidf_vectorizer.vocabulary_:
                    continue
                if args.edge_weighting == 'semantic_entropy':
                    self.add_to_graph(keyword.lower(), label.lower(), keywords_relation, keyword_entropy)
                    graph_weight_list.append(keyword_entropy)
                elif args.edge_weighting == 'tfidf':
                    tfidf_score = self.tfidf_predict(self.tfidf_vectorizer, self.tfidf_matrix, cleaned_keyword,
                                                     text_idx)
                    self.add_to_graph(keyword.lower(), label.lower(), keywords_relation, 1 - tfidf_score)

                    graph_weight_list.append(1 - tfidf_score* prediction_confidence)
                    if 1 - tfidf_score <= 0:
                        print(cleaned_keyword)
                        print(tfidf_score)
                        raise ValueError
                else:
                    self.add_to_graph(keyword.lower(), label.lower(), keywords_relation)
        # if graph_weight_list:
        # mean_weight = sum(graph_weight_list) / len(graph_weight_list)
        # for labels in G.neighbors('research area'):
        # self.Graph[label.lower()]['research area']['weight'] = mean_weight / 2
        # self.Graph[label.lower()]['research area']['total_weight'] = mean_weight / 2
        return self.Graph

    def LLM_Extraction(self, text: str, max_new_token: int, target='text',
                       delimiter='[SEP]',
                       language='en',
                       label='none'):
        if language == 'CN':
            if target == 'code':
                content = (f'从以下脚本代码中提取一些关键指令，用{delimiter}分割，'
                           f'请只给我提取出的关键指令，不要返回其他任何内容：\n{text} ')

            else:
                raise ValueError('Only implemented code extraction!')
        else:
            if target == 'text':
                if label != 'none':
                    content = (f'Extract some academic keywords from the following '
                               f"academic research paper, from {label} domain, split extracted keywords with {delimiter}, "
                               f"please only give me the keywords extracted, not any other words:\n{text} ")
                else:
                    content = (f'Extract some academic keywords from the following '
                               f"text, split extracted keywords with {delimiter}, "
                               f"please only give me the keywords extracted, not any other words:\n{text} ")
            elif target == 'desc':
                content = ('Extract some keywords from the following '
                           f"academic research area's description, split keywords with {delimiter}, "
                           f"please only give me the keywords extracted, not any other words:\n{text} ")
            elif target == 'keyword':
                content = ('Extract some keywords from the following '
                           f"description of academic research term, split keywords with {delimiter}, "
                           f"please only give me the keywords extracted, not any other words:\n{text} ")
            else:
                raise ValueError('Only implemented text or description extraction!')
        prompt = [{"role": "user", "content": content}]
        response, keyword_score = self.chat(prompt, max_new_token)
        extracted_keywords = response.split(delimiter)
        processed_keywords = []
        keyword_entropy_list = []
        skip = 0
        for keywords in extracted_keywords:
            # if args.semantic_entropy:
            if not keywords or keywords.strip() == '[]':
                continue
            else:
                processed_keywords.append(process_keyword(keywords.strip()))
        '''
        for keywords in extracted_keywords:
            # if args.semantic_entropy:
            if not keywords:
                continue
            keyword_token = self.tokenizer([keywords], return_tensors="pt", add_special_tokens=False).input_ids
            keyword_entropy = semantic_entropy(keyword_token[0], keyword_score[0], keyword_score[1])
            if keyword_entropy == -1.0:
                skip += 1
                continue
            processed_keywords.append(process_keyword(keywords))
            keyword_entropy_list.append(keyword_entropy)
        
        content = ('Give me some keywords related to the following '
                   f"academic research paper abstract, split keywords with ',', "
                   f"please only give me the related keywords:\n{text} ")
        prompt = [{"role": "user", "content": content}]
        '''
        return processed_keywords, keyword_entropy_list

    def hie_keywords_indexing(self, keyword, keywords_relation):
        keyword_extract_prompt = [{"role": "user", "content": f'What is {keyword}?'}]
        keyword_desc, res_score_hie = self.chat(keyword_extract_prompt, -1)
        keyword_list, keyword_entropy = self.LLM_Extraction(keyword_desc, -1,
                                                            target='keyword')
        for keyword_of_keyword in keyword_list:
            keyword_of_keyword = process_keyword(keyword_of_keyword)
            if not keyword_of_keyword or re.match(r'\W', keyword):
                continue
            self.add_to_graph(keyword_of_keyword.lower(), keyword.lower(), keywords_relation)

    def chat(self, message, max_new_token):
        if 'qwen' in self.LLM_name:
            text = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
                return_tensors="pt"
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.LLM_model.device)
            # print(max_new_token)
            outputs = self.LLM_model.generate(
                **model_inputs,
                max_new_tokens=max_new_token if max_new_token > 0 else 512,
                return_dict_in_generate=True,
                output_scores=True
            )
            response = outputs.sequences[0, model_inputs.input_ids.shape[-1]:]
        else:
            input_ids = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.LLM_model.device)
            # print(input_ids.shape)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.LLM_model.device)
            if self.LLM_name == 'llama3.1':
                max_token_llm = 512
            else:
                max_token_llm = None
            outputs = self.LLM_model.generate(
                input_ids,
                max_new_tokens=max_new_token if max_new_token > 0 else max_token_llm,
                eos_token_id=self.terminators,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True
            )
            # print(outputs.sequences)
            response = outputs.sequences[0, input_ids.shape[-1]:]
        # print(response)
        return self.tokenizer.decode(response, skip_special_tokens=True), (response, outputs.scores)

    def tfidf_predict(self, vectorizer, matrix, query, doc_idx):
        # feature_names = vectorizer.get_feature_names_out()
        # keyword_splitted = query.split(' ')
        # tfidf_list = []
        # tfidf_score = 1
        # for keyword_tok in keyword_splitted:
        # cleaned_query = re.findall(r'(?u)\b\w\w+\b', query)
        # cleaned_query = ' '.join(cleaned_query)
        idx = vectorizer.vocabulary_[query]
        tfidf_score = matrix.toarray()[doc_idx][idx]
        # for score in tfidf_list:
        #     tfidf_score *= score
        '''
            tfidf_matrix = vectorizer.transform([query]).todense()
            def get_feature_index(txt):
                return txt[0, :].nonzero()[1]


            result = [dict(zip([feature_names[i] for i in get_feature_index(_)], [_[0, x] for x in get_feature_index(_)])) for _
                      in tfidf_matrix]
            # result_sorted = []
            # print(result)
            # for corpus_word_list in result:
            corpus_word_list = sorted(result[0].items(), key=lambda x: x[1], reverse=True)
            # print(corpus_word_list)
            result_sorted = [x[0] for x in corpus_word_list]
            # print(result_sorted)

            return result_sorted
            '''
        if isinstance(tfidf_score, np.ndarray):
            print(tfidf_score)
            raise ValueError
        if tfidf_score > 1:
            raise ValueError
        return tfidf_score

    def generate_keywords(self, train_loader: DataLoader):
        label_keyword_dict = {}
        for data in train_loader:
            text = data['doc_token'][0]
            label = data['label'][0]
            if label not in label_keyword_dict:
                label_keyword_dict[label] = set()
            clean_text_list = re.findall(r'(?u)\b\w\w+\b', text)
            cleaned_text = str(' '.join(clean_text_list))
            text_keywords, _ = self.LLM_Extraction(cleaned_text, -1, target='text')
            for keyword in text_keywords:
                label_keyword_dict[label].add(keyword)
        for label in label_keyword_dict:
            label_keyword_dict[label] = list(label_keyword_dict[label])
        return label_keyword_dict

    def get_tfidf_vectorizer(self, corpus: list[str]):
        self.tfidf_vectorizer = TfidfVectorizer(use_idf=True,
                                                smooth_idf=True,
                                                norm='l2',
                                                analyzer='word',
                                                lowercase=True,
                                                stop_words='english',
                                                ngram_range=(1, 3)
                                                )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

    def online_indexing(self, keywords: set, responses: str, args,
                        candidate_labels: list[str], corpus, inferenced_corpus):
        keywords_list = list(keywords)
        if args.edge_weighting == 'tfidf':
            self.get_tfidf_vectorizer(corpus)
        for idx, keyword in enumerate(keywords_list):
            if args.edge_weighting == 'unit':
                self.add_to_graph(keyword.lower(), responses.lower().strip(), 'related to')
            else:
                if args.edge_weighting == 'semantic_entropy':
                    entropy_dict = self.keyword_entropy(keyword, candidate_labels, args)
                    for response_entropy in entropy_dict:
                        entropy_dict[response_entropy] = sum(entropy_dict[response_entropy]) / np.log(
                            args.keyword_inference_num)
                    for response_label, response_entropy in entropy_dict.items():
                        self.add_to_graph(keyword, response_label.lower(), 'related to', response_entropy)
                elif args.edge_weighting == 'tfidf':
                    cleaned_keyword = re.findall(r'(?u)\b\w\w+\b', keyword)
                    cleaned_keyword = ' '.join(cleaned_keyword)
                    if cleaned_keyword not in self.tfidf_vectorizer.vocabulary_:
                        continue
                    tfidf_score = self.tfidf_predict(self.tfidf_vectorizer, self.tfidf_matrix, cleaned_keyword, -1)
                    self.add_to_graph(keyword.lower().strip(), responses.lower().strip(), 'related to', 1 - tfidf_score)
