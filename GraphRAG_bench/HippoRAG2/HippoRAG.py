import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import networkx as nx
import concurrent.futures

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

# === 新增：顶层辅助函数，放在类定义之前 ===
"""
def _run_graph_retrieval(args):
    hipporag_args, query, link_top_k, query_fact_scores, top_k_facts, top_k_fact_indices = args
    self = hipporag_args['self']
    return self.graph_retrieval(query, link_top_k, query_fact_scores, top_k_facts, top_k_fact_indices)


def _run_dense_passage_retrieval(args):
    hipporag_args, query = args
    self = hipporag_args['self']
    return self.dense_passage_retrieval(query)
"""


def graph_retrieval_worker(
        query,
        link_top_k,
        query_fact_scores,
        top_k_facts,
        top_k_fact_indices,
        dpr_result,
        pagerank_dict,
        nx_graph_data,
        node_name_to_vertex_idx,
        passage_node_keys,
        fact_node_keys,
        entity_embeddings_map,
        fact_embeddings_map,
        passage_embeddings_map,
        entity_text_map,
        fact_text_map,
        passage_text_map,
        query_to_embedding,
        edge_to_relation,
        config
):
    nx_graph = nx.node_link_graph(nx_graph_data, edges="links")
    nx.set_node_attributes(nx_graph, pagerank_dict, 'pagerank')
    query_emb = query_to_embedding['triple'][query]
    # 预处理 fact id 到 (source, target, 原文三元组) 的映射
    fact_hash_to_tuple = {}
    tuple_to_fact_id = {}
    tuple_to_fact = {}
    for fact_id in fact_node_keys:
        fact_row = fact_text_map[fact_id]
        fact = eval(fact_row['content'])  # (subject, predicate, object)
        source_hash = compute_mdhash_id(fact[0], prefix="entity-")
        target_hash = compute_mdhash_id(fact[2], prefix="entity-")
        fact_hash_to_tuple[fact_id] = (source_hash, target_hash)
        tuple_to_fact_id[(source_hash, target_hash)] = fact_id
        tuple_to_fact_id[(target_hash, source_hash)] = fact_id
        tuple_to_fact[(source_hash, target_hash)] = fact
        tuple_to_fact[(target_hash, source_hash)] = (fact[2], fact[1], fact[0])
    dpr_sorted_doc_ids, dpr_sorted_doc_scores = dpr_result
    dpr_score_map = {passage_node_keys[idx]: float(score) for idx, score in
                     zip(dpr_sorted_doc_ids, dpr_sorted_doc_scores)}
    edge_costs = {}
    for source, target, edge_data in nx_graph.edges(data=True):
        relation = edge_data.get('relation', 'related_to')
        cost = 0.5
        if relation == 'passage_connector':
            cost = 10.0
        elif relation not in ['synonym', 'passage_link', 'passage_connector']:
            fact_id = tuple_to_fact_id.get((source, target), None)
            fact = tuple_to_fact.get((source, target), None)
            if fact_id and fact is not None:
                fact_emb = fact_embeddings_map.get(fact_id, None)
                if fact_emb is not None:
                    similarity = float(np.dot(fact_emb, query_emb))
                    cost = 1 - similarity
        elif relation == 'synonym':
            emb1 = entity_embeddings_map.get(source, None)
            emb2 = entity_embeddings_map.get(target, None)
            if emb1 is not None and emb2 is not None:
                sim1 = float(np.dot(emb1, query_emb))
                sim2 = float(np.dot(emb2, query_emb))
                similarity = max(sim1, sim2)
                cost = 1 - similarity
        elif relation == 'passage_link':
            passage_node = source if source in passage_node_keys else target
            similarity = dpr_score_map.get(passage_node, 0.0)
            cost = 1 - similarity
        edge_costs[(source, target)] = cost
        edge_costs[(target, source)] = cost
    for (u, v), cost in edge_costs.items():
        if nx_graph.has_edge(u, v):
            nx_graph[u][v]['weight'] = cost
    # 4. 使用link_top_k筛选top facts，只保留前link_top_k个fact的anchor节点
    if link_top_k and len(top_k_facts) > link_top_k:
        fact_to_score = {}
        for f in top_k_facts:
            source_hash = compute_mdhash_id(f[0], prefix="entity-")
            target_hash = compute_mdhash_id(f[2], prefix="entity-")
            fact_id = tuple_to_fact_id.get((source_hash, target_hash), None)
            if fact_id:
                fact_emb = fact_embeddings_map.get(fact_id, None)
                if fact_emb is not None:
                    similarity = float(np.dot(fact_emb, query_emb))
                    fact_to_score[f] = similarity
                else:
                    fact_to_score[f] = 0.0
            else:
                fact_to_score[f] = 0.0
        sorted_facts = sorted(top_k_facts, key=lambda f: fact_to_score.get(f, 0), reverse=True)
        top_facts = sorted_facts[:link_top_k]
    else:
        top_facts = top_k_facts
    anchor_nodes = set()
    for f in top_facts:
        anchor_nodes.add(compute_mdhash_id(f[0], prefix="entity-"))
        anchor_nodes.add(compute_mdhash_id(f[2], prefix="entity-"))
    try:
        steiner_tree = nx.algorithms.approximation.steiner_tree(nx_graph, anchor_nodes, weight='weight')
    except Exception as e:
        return [], []
    subgraph_nodes = set(steiner_tree.nodes())
    subgraph_edges = set(steiner_tree.edges())
    iteration = 0
    threshold = 20
    while iteration < threshold:
        iteration += 1
        # Compute r_MIMC as before
        ratio_list = []
        for edge in subgraph_edges:
            source, target = edge
            if (source, target) in edge_costs:
                sum_score = nx_graph.nodes[source].get('pagerank', 0) + nx_graph.nodes[target].get('pagerank', 0)
                ratio_list.append(sum_score / 2 * edge_costs[(source, target)])
        # avg_pagerank = np.mean([nx_graph.nodes[n].get('pagerank', 0) for n in subgraph_nodes])
        r_MIMC = np.mean(ratio_list) if ratio_list else 0
        neighbors = set()
        for n in subgraph_nodes:
            for nei in nx_graph.neighbors(n):
                if nei not in subgraph_nodes:
                    neighbors.add(nei)
        if not neighbors:
            break
        # Compute score/cost ratio for each neighbor
        # neighbor_scores = [(nei, nx_graph.nodes[nei].get('pagerank', 0)) for nei in neighbors]
        # max_neighbor_score = max([s for _, s in neighbor_scores]) if neighbor_scores else 0
        neighbor_ratios = []
        for nei in neighbors:
            score = nx_graph.nodes[nei].get('pagerank', 0)
            min_cost = min([nx_graph[n][nei]['weight'] for n in subgraph_nodes if nx_graph.has_edge(n, nei)])
            if min_cost > 0:
                neighbor_ratios.append(score / min_cost)
            else:
                neighbor_ratios.append(float('inf'))
        max_neighbor_ratio = max(neighbor_ratios) if neighbor_ratios else 0
        if max_neighbor_ratio < r_MIMC:
            break
        # Select best neighbor as before
        best_nei = None
        best_ratio = -1
        best_score = -1
        best_cost = float('inf')
        for nei in neighbors:
            score = nx_graph.nodes[nei].get('pagerank', 0)
            min_cost = min([nx_graph[n][nei]['weight'] for n in subgraph_nodes if nx_graph.has_edge(n, nei)])
            ratio = score / min_cost if min_cost > 0 else float('inf')
            if ratio > best_ratio or (
                    ratio == best_ratio and (score > best_score or (score == best_score and min_cost < best_cost))):
                best_ratio = ratio
                best_score = score
                best_cost = min_cost
                best_nei = nei
        if best_nei is not None:
            subgraph_nodes.add(best_nei)
        else:
            break
    entity_map = entity_text_map
    subgraph = nx_graph.subgraph(subgraph_nodes)
    subgraph_strings = []
    edges = list(subgraph.edges(data=True))
    passage_nodes_on_path = []
    if edges:
        components = list(nx.connected_components(subgraph))
        for component in components:
            if len(component) < 2:
                continue
            component_subgraph = subgraph.subgraph(component)
            component_edges = list(component_subgraph.edges(data=True))
            edge_strings = []
            for source, target, edge_data in component_edges:
                if (source in passage_node_keys or target in passage_node_keys):
                    if source in passage_node_keys:
                        passage_nodes_on_path.append(source)
                    if target in passage_node_keys:
                        passage_nodes_on_path.append(target)
                    continue
                # 用真实谓词和实体名
                fact = tuple_to_fact.get((source, target), None)
                if fact is not None:
                    source_name = fact[0]
                    target_name = fact[2]
                    relation = fact[1]
                else:
                    # fallback: 用 embedding store 原文
                    source_name = entity_map.get(source, {}).get('content', source)
                    target_name = entity_map.get(target, {}).get('content', target)
                    relation = edge_data.get('relation', 'related_to')
                edge_strings.append(f"({source_name} [SEP] {relation} [SEP] {target_name})")
            if edge_strings:
                subgraph_strings.append(" [SEP_REC] ".join(edge_strings))
    passage_nodes_on_path = list({k for k in passage_nodes_on_path})
    passage_texts = [passage_text_map.get(k, {}).get('content', k) for k in passage_nodes_on_path]
    return subgraph_strings, passage_texts


class HippoRAG:

    def __init__(self, global_config=None, save_dir=None, llm_model_name=None, embedding_model_name=None,
                 llm_base_url=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific HippoRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed vLLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        # LLM and embedding model specific working directories are created under every specified saving directories

        # if '7B-Instruct' in llm_label:
        #    llm_label = llm_label.replace('7B', '14B')
        if 'Qwen-2.5-7B-Base-RAG-RL'.lower() in self.global_config.llm_name.lower():
            llm_label = 'Qwen_Qwen2.5-14B-Instruct'
        else:
            llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        if 'Qwen-2.5-7B-Base-RAG-RL'.lower() in self.global_config.llm_name.lower():
            self.openie_results_path = os.path.join(self.global_config.save_dir,
                                                    f'openie_results_ner_Qwen_Qwen2.5-14B-Instruct.json')
        else:
            self.openie_results_path = os.path.join(self.global_config.save_dir,
                                                    f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)
        self.edge_to_relation = {}
        self.synonym_edges = set()
        self.llm_token_usage = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})

        self.ready_to_retrieve = False

    def initialize_graph(self):
        """
        Initializes a graph using a GraphML file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a GraphML file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graphml_xml_file = os.path.join(
            self.working_dir, f"graph.graphml"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graphml_xml_file):
                preloaded_graph = ig.Graph.Read_GraphML(self._graphml_xml_file)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self, docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k: chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunks = self.chunk_embedding_store.get_text_for_all_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k: chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            # 传递 HippoRAG 实例用于 token 统计
            self.openie.hipporag_ref = self
            if self.global_config.rag_mode == 'gorag':
                new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie_tfidf(new_openie_rows,
                                                                                               max_ngram_length=self.global_config.max_ngram_length,
                                                                                               tfidf_threshold=self.global_config.ner_threshold)
            else:
                new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunks) == len(ner_results_dict) == len(triple_results_dict)

        # prepare data_store
        chunk_ids = list(chunks.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_num_chunk = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

        # 初始化 node keys，保证 networkx 图构建时可用
        self.entity_node_keys = list(self.entity_embedding_store.get_all_ids())
        self.passage_node_keys = list(self.chunk_embedding_store.get_all_ids())
        self.fact_node_keys = list(self.fact_embedding_store.get_all_ids())

        # 构造并缓存 networkx 图
        nx_graph = nx.Graph()
        for i, node_name in enumerate(self.graph.vs['name']):
            nx_graph.add_node(node_name)
        for edge in self.graph.es:
            source_name = self.graph.vs[edge.source]['name']
            target_name = self.graph.vs[edge.target]['name']
            relation = edge['relation'] if 'relation' in edge.attributes() else 'related_to'
            nx_graph.add_edge(source_name, target_name, relation=relation)
        # 添加 pseudonode 及 passage_connector 边
        pseudonode = 'pseudo_passage_connector'
        nx_graph.add_node(pseudonode)
        for passage_node in self.passage_node_keys:
            nx_graph.add_edge(passage_node, pseudonode, relation='passage_connector', weight=10.0)
        self.nx_graph = nx_graph

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []
        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def retrieve_gorag(self,
                       queries: List[str],
                       num_to_retrieve: int = None,
                       gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = [None] * len(queries)
        max_workers = 16  # 可根据硬件调整

        # 1. 预处理：计算fact分数和rerank
        precomputed_data = [None] * len(queries)
        for q_idx, query in tqdm(enumerate(queries), desc="Preprocess", total=len(queries)):
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            precomputed_data[q_idx] = (query, query_fact_scores, top_k_fact_indices, top_k_facts, rerank_log)

        # 2. 并行DPR
        dpr_results = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            dpr_futures = {executor.submit(self.dense_passage_retrieval, queries[i]): i for i in range(len(queries))}
            for future in tqdm(as_completed(dpr_futures), total=len(queries), desc="DPR"):
                i = dpr_futures[future]
                try:
                    sorted_doc_ids, sorted_doc_scores = future.result()
                except Exception as e:
                    logger.error(f"Error in dense_passage_retrieval for query idx {i}: {e}")
                    sorted_doc_ids, sorted_doc_scores = [], []
                dpr_results[i] = (sorted_doc_ids, sorted_doc_scores)

        # 3. 根据fact情况分流：有fact的进graph_retrieval，无fact的直接用DPR
        graph_task_indices = []
        graph_task_params = []
        for i, (query, query_fact_scores, top_k_fact_indices, top_k_facts, rerank_log) in enumerate(precomputed_data):
            sorted_doc_ids, sorted_doc_scores = dpr_results[i]
            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]
            if len(top_k_facts) == 0:
                doc_scores = sorted_doc_scores[:num_to_retrieve] if sorted_doc_scores is not None else np.array(
                    [0.0] * len(top_k_docs))
                retrieval_results[i] = QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=doc_scores,
                    generated_graph='False',
                    passage_nodes_on_path=None
                )
            else:
                link_top_k = self.global_config.linking_top_k
                personalized_nodes = set()
                for f in top_k_facts:
                    personalized_nodes.add(compute_mdhash_id(f[0], prefix="entity-"))
                    personalized_nodes.add(compute_mdhash_id(f[2], prefix="entity-"))
                reset_vec = np.zeros(len(self.graph.vs))
                for node in personalized_nodes:
                    idx = self.node_name_to_vertex_idx.get(node, None)
                    if idx is not None:
                        reset_vec[idx] = 1.0
                if reset_vec.sum() > 0:
                    reset_vec = reset_vec / reset_vec.sum()
                else:
                    reset_vec = None
                damping = self.global_config.damping or 0.5
                pagerank_scores = self.graph.personalized_pagerank(
                    vertices=range(len(self.node_name_to_vertex_idx)),
                    damping=damping,
                    directed=False,
                    weights='weight',
                    reset=reset_vec,
                    implementation='prpack'
                )
                pagerank_dict = {self.graph.vs[idx]['name']: score for idx, score in enumerate(pagerank_scores)}
                # --- 新增：embedding 映射 ---
                entity_node_keys = list(self.entity_embedding_store.get_all_ids())
                entity_embeddings = self.entity_embeddings
                entity_embeddings_map = {k: v for k, v in zip(entity_node_keys, entity_embeddings)}
                fact_node_keys = list(self.fact_embedding_store.get_all_ids())
                fact_embeddings = self.fact_embeddings
                fact_embeddings_map = {k: v for k, v in zip(fact_node_keys, fact_embeddings)}
                passage_node_keys = list(self.chunk_embedding_store.get_all_ids())
                passage_embeddings = self.passage_embeddings
                passage_embeddings_map = {k: v for k, v in zip(passage_node_keys, passage_embeddings)}
                edge_to_relation = self.edge_to_relation
                graph_task_indices.append(i)
                graph_task_params.append((query, link_top_k, query_fact_scores, top_k_facts, top_k_fact_indices,
                                          dpr_results[i], pagerank_dict, nx.node_link_data(self.nx_graph),
                                          self.node_name_to_vertex_idx, passage_node_keys, fact_node_keys,
                                          entity_embeddings_map, fact_embeddings_map, passage_embeddings_map,
                                          self.entity_embedding_store.get_text_for_all_rows(),
                                          self.fact_embedding_store.get_text_for_all_rows(),
                                          self.chunk_embedding_store.get_text_for_all_rows(), self.query_to_embedding,
                                          edge_to_relation, {"damping": self.global_config.damping,
                                                             "linking_top_k": self.global_config.linking_top_k,
                                                             "graph_summary": self.global_config.graph_summary}))

        if graph_task_params:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                graph_futures = {executor.submit(graph_retrieval_worker, *params): idx for idx, params in
                                 zip(graph_task_indices, graph_task_params)}
                for future in tqdm(as_completed(graph_futures), total=len(graph_futures), desc="Graph Retrieval"):
                    idx = graph_futures[future]
                    query, query_fact_scores, top_k_fact_indices, top_k_facts, rerank_log = precomputed_data[idx]
                    try:
                        subgraph_strings, passage_texts = future.result()
                    except Exception as e:
                        logger.error(f"Error in graph_retrieval for query idx {idx}: {e}")
                        subgraph_strings, passage_texts = [], None
                    sorted_doc_ids, sorted_doc_scores = dpr_results[idx]
                    if subgraph_strings is None:
                        subgraph_strings = []
                    if passage_texts is None:
                        passage_texts = []
                    top_k_docs = list(subgraph_strings)  # 先用 subgraph_strings
                    generated_graph = 'True'
                    # 新增：补充逻辑
                    total_count = len(top_k_docs) + len(passage_texts)
                    link_top_k = self.global_config.linking_top_k
                    if total_count < link_top_k:
                        needed = link_top_k - total_count
                        # passage_texts 已有的内容
                        existing_passages = set(passage_texts)
                        # 按照 dpr 检索顺序补充
                        for idx_ in sorted_doc_ids:
                            passage = self.chunk_embedding_store.get_row(self.passage_node_keys[idx_])["content"]
                            if passage not in existing_passages and passage not in top_k_docs:
                                top_k_docs.append(passage)
                                existing_passages.add(passage)
                                needed -= 1
                                if needed == 0:
                                    break
                    if not top_k_docs:
                        top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[i])["content"] for i in
                                      sorted_doc_ids[:num_to_retrieve]]
                        generated_graph = 'False'
                    if self.global_config.graph_summary:
                        top_k_docs = self.summarize_subgraphs_with_llm(top_k_docs, query)
                    if sorted_doc_scores is not None and generated_graph == 'False':
                        doc_scores = sorted_doc_scores[:num_to_retrieve]
                    else:
                        doc_scores = np.array([0.0] * len(top_k_docs))
                    retrieval_results[idx] = QuerySolution(
                        question=query,
                        docs=top_k_docs,
                        doc_scores=doc_scores,
                        generated_graph=generated_graph,
                        passage_nodes_on_path=passage_texts
                    )
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str | QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[
        List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the HippoRAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if self.global_config.rag_mode == 'gorag':
                if gold_docs is not None:
                    queries, overall_retrieval_result = self.retrieve_gorag(queries=queries, gold_docs=gold_docs)
                else:
                    queries = self.retrieve_gorag(queries=queries)
            else:
                if gold_docs is not None:
                    queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
                else:
                    queries = self.retrieve(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        # Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            # gorag模式下拼接passage_nodes_on_path
            if self.global_config.rag_mode == 'gorag' and self.global_config.include_passage_nodes_in_qa_input:
                if hasattr(query_solution, 'passage_nodes_on_path') and query_solution.passage_nodes_on_path:
                    prompt_user += '\n----\nReference Passages:\n' + '\n'.join(
                        query_solution.passage_nodes_on_path) + '\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        # Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                answer = response_content.split('Answer:')
                if len(answer) > 1:
                    pred_ans = answer[1].strip()
                else:
                    pred_ans = answer[0].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!\n"
                               f"LLM response:{response_content}")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        # 统计 token
        for meta in all_metadata:
            if isinstance(meta, dict) and 'prompt_tokens' in meta and 'completion_tokens' in meta:
                self.llm_token_usage['qa']['prompt_tokens'] += meta['prompt_tokens']
                self.llm_token_usage['qa']['completion_tokens'] += meta['completion_tokens']
                self.llm_token_usage['qa']['calls'] += 1

        return queries_solutions, all_response_message, all_metadata

    def ensure_nodes_exist(self, node_names: List[str]):
        """
        Helper to ensure all nodes in node_names exist in the graph (by 'name' attribute).
        """
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()
        new_nodes = list(set(node_names) - current_graph_nodes)
        if new_nodes:
            self.graph.add_vertices(n=len(new_nodes), attributes={"name": new_nodes})

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.
        """
        logger.info(f"Adding OpenIE triples to graph.")
        # 1. 先收集所有需要添加的 entity 节点和 passage 节点
        all_entity_nodes = set()
        all_chunk_nodes = set(chunk_ids)
        for triples in chunk_triples:
            for triple in triples:
                triple = tuple(triple)
                node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))
                all_entity_nodes.add(node_key)
                all_entity_nodes.add(node_2_key)
        self.ensure_nodes_exist(list(all_entity_nodes | all_chunk_nodes))
        # 2. 正常添加边
        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()
            valid_edges = []
            valid_weights = {"weight": []}
            valid_relations = []
            for triple in triples:
                triple = tuple(triple)
                fact_key = compute_mdhash_id(content=str(triple), prefix=("fact-"))
                node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))
                self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                    (node_key, node_2_key), 0.0) + 1
                self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                    (node_2_key, node_key), 0.0) + 1
                # 记录原始谓词，为空时 fallback
                relation_value = triple[1] if triple[1] not in (None, '') else 'related_to'
                self.edge_to_relation[(node_key, node_2_key)] = relation_value
                self.edge_to_relation[(node_2_key, node_key)] = relation_value
                entities_in_chunk.add(node_key)
                entities_in_chunk.add(node_2_key)
                valid_edges.append((node_key, node_2_key))
                valid_weights["weight"].append(1.0)
                valid_relations.append(relation_value)
            for node in entities_in_chunk:
                self.ent_node_to_num_chunk[node] = self.ent_node_to_num_chunk.get(node, 0) + 1
            if valid_edges:
                valid_attributes = valid_weights.copy()
                valid_attributes['relation'] = valid_relations
                self.graph.add_edges(valid_edges, attributes=valid_attributes)

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.
        """
        all_entity_nodes = set()
        all_chunk_nodes = set(chunk_ids)
        for ents in chunk_triple_entities:
            for ent in ents:
                all_entity_nodes.add(compute_mdhash_id(ent, prefix="entity-"))
        self.ensure_nodes_exist(list(all_entity_nodes | all_chunk_nodes))
        num_new_chunks = 0
        logger.info(f"Connecting passage nodes to phrase nodes.")
        for idx, chunk_key in tqdm(enumerate(chunk_ids)):
            valid_edges = []
            valid_weights = {"weight": []}
            valid_relations = []
            for chunk_ent in chunk_triple_entities[idx]:
                node_key = compute_mdhash_id(chunk_ent, prefix="entity-")
                self.node_to_node_stats[(chunk_key, node_key)] = 1.0
                valid_edges.append((chunk_key, node_key))
                valid_weights["weight"].append(1.0)
                valid_relations.append('passage_link')
            if valid_edges:
                valid_attributes = valid_weights.copy()
                valid_attributes['relation'] = valid_relations
                self.graph.add_edges(valid_edges, attributes=valid_attributes)
            num_new_chunks += 1
        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")
        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())
        self.ensure_nodes_exist(entity_node_keys)
        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)
        num_synonym_triple = 0
        synonym_candidates = []
        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []
            entity = self.entity_id_to_row[node_key]["content"]
            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]
                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break
                    nn_phrase = self.entity_id_to_row[nn]["content"]
                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1
                        self.node_to_node_stats[sim_edge] = score
                        # 记录 synonym 边
                        self.synonym_edges.add((node_key, nn))
                        self.synonym_edges.add((nn, node_key))
                        if not self.graph.are_connected(node_key, nn):
                            self.ensure_nodes_exist([node_key, nn])
                            self.graph.add_edge(node_key, nn, weight=score, relation='synonym')
                        num_nns += 1
            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            # Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            try:
                openie_dict = {'docs': all_openie_info, 'avg_ent_chars': round(sum_phrase_chars / num_phrases, 4),
                               'avg_ent_words': round(sum_phrase_words / num_phrases, 4)}
            except ZeroDivisionError:
                openie_dict = {'docs': all_openie_info, 'avg_ent_chars': 0.0, 'avg_ent_words': 0.0}

            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """
        entity_nodes = self.entity_embedding_store.get_text_for_all_rows()
        passage_nodes = self.chunk_embedding_store.get_text_for_all_rows()
        all_nodes = list(entity_nodes.keys()) + list(passage_nodes.keys())
        self.ensure_nodes_exist(all_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        # 获取 passage node keys，避免 self.passage_node_keys 未初始化的问题
        passage_node_keys = set(self.chunk_embedding_store.get_all_ids())

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            # 优先用 edge_to_relation 查 relation
            relation = self.edge_to_relation.get((edge[0], edge[1]), None)
            if relation is None or relation == '':
                if (edge[0], edge[1]) in self.synonym_edges or (edge[1], edge[0]) in self.synonym_edges:
                    relation = 'synonym'
                elif edge[0] in passage_node_keys or edge[1] in passage_node_keys:
                    relation = 'passage_link'
                else:
                    relation = 'related_to'
            edge_metadata.append({
                "weight": weight,
                "relation": relation
            })

        valid_edges, valid_weights, valid_relations = [], {"weight": []}, []
        self.ensure_nodes_exist(edge_source_node_keys + edge_target_node_keys)
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                relation = edge_d.get("relation", "related_to")
                valid_weights["weight"].append(weight)
                valid_relations.append(relation)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        if valid_edges:
            valid_attributes = valid_weights.copy()
            valid_attributes['relation'] = valid_relations
            self.graph.add_edges(
                valid_edges,
                attributes=valid_attributes
            )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_graphml(self._graphml_xml_file)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids())  # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids())  # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        assert len(self.entity_node_keys) + len(self.passage_node_keys) == self.graph.vcount()

        igraph_name_to_idx = {node["name"]: idx for idx, node in
                              enumerate(self.graph.vs)}  # from node key to the index in the backbone graph
        self.node_name_to_vertex_idx = igraph_name_to_idx
        self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in
                                 self.entity_node_keys]  # a list of backbone graph node index
        self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in
                                  self.passage_node_keys]  # a list of backbone passage node index

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction(
                                                                                'query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction(
                                                                                 'query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T)  # shape: (#facts, )
        query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
        query_fact_scores = min_max_normalize(query_fact_scores)

        return query_fact_scores

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        if self.global_config.retrieval_mode == 'hybrid':
            # Perform BM25 retrieval for ALL documents
            bm25_scores = self.chunk_embedding_store.bm25_retrieve_all(query)

            if len(bm25_scores) > 0 and np.max(bm25_scores) > 0:
                # Normalize BM25 scores
                bm25_scores = min_max_normalize(bm25_scores)

                # Combine dense and sparse scores (average)
                # Both query_doc_scores and bm25_scores have the same length and order
                hybrid_scores = (query_doc_scores + bm25_scores) / 2.0

                query_doc_scores = hybrid_scores

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_retrieval(self,
                        query: str,
                        link_top_k: int,
                        query_fact_scores: np.ndarray,
                        top_k_facts: List[Tuple],
                        top_k_fact_indices: List[str],
                        dpr_result,
                        threshold: int = 20
                        ):
        # logger.warning(f"[graph_retrieval] Start for query: {query}")
        # 1. 直接用缓存的 networkx 图
        nx_graph = self.nx_graph.copy()
        # logger.warning(f"[graph_retrieval] nx_graph nodes: {len(nx_graph.nodes)}, edges: {len(nx_graph.edges)}")
        # 2. 用 igraph 计算 personalized PageRank (PPR) 分数
        query_emb = self.query_to_embedding['triple'][query]
        # 构造 personalized reset 向量（如只对 anchor nodes 或 top_k_facts 相关节点赋 1，其余为 0）
        reset_vec = np.zeros(len(self.graph.vs))
        # 以 top_k_facts 的 subject/object 节点为 personalized 起点
        personalized_nodes = set()
        for f in top_k_facts:
            personalized_nodes.add(compute_mdhash_id(f[0], prefix="entity-"))
            personalized_nodes.add(compute_mdhash_id(f[2], prefix="entity-"))
        logger.warning(f"[graph_retrieval] personalized_nodes: {len(personalized_nodes)}")
        for node in personalized_nodes:
            idx = self.node_name_to_vertex_idx.get(node, None)
            if idx is not None:
                reset_vec[idx] = 1.0
        if reset_vec.sum() > 0:
            reset_vec = reset_vec / reset_vec.sum()
        else:
            reset_vec = None  # fallback to uniform
        if self.global_config.damping is None:
            damping = 0.5
        else:
            damping = self.global_config.damping
        # logger.warning("[graph_retrieval] Before personalized PageRank")
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_vec,
            implementation='prpack'
        )
        logger.warning("[graph_retrieval] After personalized PageRank")
        pagerank_dict = {self.graph.vs[idx]['name']: score for idx, score in enumerate(pagerank_scores)}
        nx.set_node_attributes(nx_graph, pagerank_dict, 'pagerank')

        # 3. 只在需要时取 embedding 计算边权
        # 预处理 fact id 到 (source, target) 的映射
        fact_hash_to_tuple = {}
        tuple_to_fact_id = {}
        for fact_id in self.fact_node_keys:
            fact_row = self.fact_embedding_store.get_row(fact_id)
            fact = eval(fact_row['content'])  # (subject, predicate, object)
            source_hash = compute_mdhash_id(fact[0], prefix="entity-")
            target_hash = compute_mdhash_id(fact[2], prefix="entity-")
            fact_hash_to_tuple[fact_id] = (source_hash, target_hash)
            tuple_to_fact_id[(source_hash, target_hash)] = fact_id
            tuple_to_fact_id[(target_hash, source_hash)] = fact_id

        # passage DPR 分数（用传入的dpr_result）
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = dpr_result
        dpr_score_map = {self.passage_node_keys[idx]: float(score) for idx, score in
                         zip(dpr_sorted_doc_ids, dpr_sorted_doc_scores)}

        edge_costs = {}
        for source, target, edge_data in nx_graph.edges(data=True):
            relation = edge_data.get('relation', 'related_to')
            cost = 0.5  # 默认cost
            if relation == 'passage_connector':
                cost = 10.0
            elif relation not in ['synonym', 'passage_link', 'passage_connector']:
                # fact triple边
                fact_id = tuple_to_fact_id.get((source, target), None)
                if fact_id:
                    fact_emb = self.fact_embedding_store.get_embeddings([fact_id])[0]
                    similarity = float(np.dot(fact_emb, query_emb))
                    cost = 1 - similarity
            elif relation == 'synonym':
                entity_embs = self.entity_embedding_store.get_embeddings([source, target])
                sim1 = float(np.dot(entity_embs[0], query_emb))
                sim2 = float(np.dot(entity_embs[1], query_emb))
                similarity = max(sim1, sim2)
                cost = 1 - similarity
            elif relation == 'passage_link':
                passage_node = source if source in self.passage_node_keys else target
                similarity = dpr_score_map.get(passage_node, 0.0)
                cost = 1 - similarity
            edge_costs[(source, target)] = cost
            edge_costs[(target, source)] = cost
        # 重新赋予边权
        for (u, v), cost in edge_costs.items():
            if nx_graph.has_edge(u, v):
                nx_graph[u][v]['weight'] = cost

        # 4. 使用link_top_k筛选top facts，只保留前link_top_k个fact的anchor节点
        if link_top_k and len(top_k_facts) > link_top_k:
            # 只在需要时计算 fact_to_score
            fact_to_score = {}
            for f in top_k_facts:
                fact_id = tuple_to_fact_id.get(
                    (compute_mdhash_id(f[0], prefix="entity-"), compute_mdhash_id(f[2], prefix="entity-")), None)
                if fact_id:
                    fact_emb = self.fact_embedding_store.get_embeddings([fact_id])[0]
                    similarity = float(np.dot(fact_emb, query_emb))
                    fact_to_score[f] = similarity
                else:
                    fact_to_score[f] = 0.0
            sorted_facts = sorted(top_k_facts, key=lambda f: fact_to_score.get(f, 0), reverse=True)
            top_facts = sorted_facts[:link_top_k]
        else:
            top_facts = top_k_facts

        # 5. anchor triple的subject/object节点（修正：用hash后的节点名）
        anchor_nodes = set()
        for f in top_facts:
            anchor_nodes.add(compute_mdhash_id(f[0], prefix="entity-"))
            anchor_nodes.add(compute_mdhash_id(f[2], prefix="entity-"))
        logger.warning(f"[graph_retrieval] anchor_nodes: {len(anchor_nodes)}")
        # 6. Steiner tree近似算法
        logger.warning("[graph_retrieval] Before Steiner tree")
        try:
            steiner_tree = nx.algorithms.approximation.steiner_tree(nx_graph, anchor_nodes, weight='weight')
        except Exception as e:
            logger.error(f"Error generating Steiner tree: {e}")
            print(f"Error generating Steiner tree: {e}")
            print('NO STEINER TREE GENERATED!!!')
            return [], []
        logger.warning("[graph_retrieval] After Steiner tree")
        subgraph_nodes = set(steiner_tree.nodes())
        # 7. 迭代扩展
        logger.warning("[graph_retrieval] Before iterative expansion")
        iteration = 0
        while iteration < threshold:
            iteration += 1
            avg_pagerank = np.mean([nx_graph.nodes[n].get('pagerank', 0) for n in subgraph_nodes])
            neighbors = set()
            for n in subgraph_nodes:
                for nei in nx_graph.neighbors(n):
                    if nei not in subgraph_nodes:
                        neighbors.add(nei)
            if not neighbors:
                break
            neighbor_scores = [(nei, nx_graph.nodes[nei].get('pagerank', 0)) for nei in neighbors]
            max_neighbor_score = max([s for _, s in neighbor_scores]) if neighbor_scores else 0
            if max_neighbor_score < avg_pagerank:
                break
            best_nei = None
            best_score = -1
            best_cost = float('inf')
            for nei, score in neighbor_scores:
                min_cost = min([nx_graph[n][nei]['weight'] for n in subgraph_nodes if nx_graph.has_edge(n, nei)])
                if score > best_score or (score == best_score and min_cost < best_cost):
                    best_score = score
                    best_cost = min_cost
                    best_nei = nei
            if best_nei is not None:
                subgraph_nodes.add(best_nei)
            else:
                break
        logger.warning(f"[graph_retrieval] After iterative expansion, total subgraph_nodes: {len(subgraph_nodes)}")
        # 8. 返回子图字符串（跳过passage节点）
        # 构建 entity hash -> 原文映射
        entity_map = self.entity_embedding_store.get_text_for_all_rows()
        subgraph = nx_graph.subgraph(subgraph_nodes)
        subgraph_strings = []
        edges = list(subgraph.edges(data=True))
        passage_nodes_on_path = []
        if edges:
            components = list(nx.connected_components(subgraph))
            for component in components:
                if len(component) < 2:
                    continue
                component_subgraph = subgraph.subgraph(component)
                component_edges = list(component_subgraph.edges(data=True))
                edge_strings = []
                for source, target, edge_data in component_edges:
                    if (source in self.passage_node_keys or target in self.passage_node_keys):
                        # 收集passage node
                        if source in self.passage_node_keys:
                            passage_nodes_on_path.append(source)
                        if target in self.passage_node_keys:
                            passage_nodes_on_path.append(target)
                        continue
                    # hash -> 原文
                    source_name = entity_map.get(source, {}).get('content', source)
                    target_name = entity_map.get(target, {}).get('content', target)
                    relation = edge_data.get('relation', 'related_to')
                    edge_strings.append(f"({source_name} [SEP] {target_name} [SEP] {relation})")
                if edge_strings:
                    subgraph_strings.append(" [SEP_REC] ".join(edge_strings))
        # passage_nodes_on_path去重并转为原文
        passage_nodes_on_path = list({k for k in passage_nodes_on_path})
        passage_text_map = self.chunk_embedding_store.get_text_for_all_rows()
        passage_texts = [passage_text_map.get(k, {}).get('content', k) for k in passage_nodes_on_path]
        return subgraph_strings, passage_texts

    def summarize_subgraphs_with_llm(self, subgraph_strings: list, query: str, prompt_template: str = None) -> list:
        """
        Use deployed LLM to summarize each subgraph string in English.
        Args:
            subgraph_strings: List[str], each is a subgraph string from graph_retrieval
            prompt_template: Optional custom prompt template with {subgraph} placeholder
        Returns:
            List[str], each is the LLM-generated summary in English
        """
        summaries = []
        subgraph = '\n'.join(subgraph_strings)
        if prompt_template is None:
            prompt_template = (
                "Below is a set of knowledge graph triples, each in the format: (Entity1 [SEP] Entity2 [SEP] Relation). "
                f"Please concisely summarize the main content of this subgraph in English, in order to answer this question: {query}"
                f"\nHere is the set of knowledge graph triples: {subgraph}\nSummary:"
            )
        for subgraph in subgraph_strings:
            prompt = prompt_template.format(subgraph=subgraph)
            messages = [{"role": "user", "content": prompt}]
            response, metadata = self.llm_model.infer(messages)
            if isinstance(response, list):
                summary = response[0]["content"] if isinstance(response[0], dict) and "content" in response[0] else str(
                    response[0])
            else:
                summary = str(response)
            summaries.append(summary.strip())
            # 补充 token 统计
            if hasattr(self, 'llm_token_usage') and 'prompt_tokens' in metadata and 'completion_tokens' in metadata:
                self.llm_token_usage['summarize']['prompt_tokens'] += metadata['prompt_tokens']
                self.llm_token_usage['summarize']['completion_tokens'] += metadata['completion_tokens']
                self.llm_token_usage['summarize']['calls'] += 1
        return summaries

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        # Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score

                    if self.ent_node_to_num_chunk[phrase_key] != 0:
                        phrase_weights[phrase_id] /= self.ent_node_to_num_chunk[phrase_key]

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                       phrase_weights,
                                                                       linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        # Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        # Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        # Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k

        candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][
                                 ::-1].tolist()  # list of ranked link_top_k fact relative indices
        real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in
                                   candidate_fact_indices]  # list of ranked link_top_k fact keys
        fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
        candidate_facts = [eval(fact_row_dict[id]['content']) for id in
                           real_candidate_fact_ids]  # list of link_top_k facts (each fact is a relation triple in tuple data type)

        # if self.global_config.rag_mode == 'gorag':
        #    return candidate_fact_indices, candidate_facts, {}
        # else:
        top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                            candidate_facts,
                                                                            candidate_fact_indices,
                                                                            len_after_rerank=link_top_k)

        rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}

        return top_k_fact_indices, top_k_facts, rerank_log

    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5  # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

    def get_llm_token_usage_summary(self):
        lines = []
        total_prompt = 0
        total_completion = 0
        total_calls = 0
        for step, usage in self.llm_token_usage.items():
            lines.append(
                f"{step}: prompt={usage['prompt_tokens']}, completion={usage['completion_tokens']}, calls={usage['calls']}")
            total_prompt += usage['prompt_tokens']
            total_completion += usage['completion_tokens']
            total_calls += usage['calls']
        lines.append(f"Total prompt tokens: {total_prompt}")
        lines.append(f"Total completion tokens: {total_completion}")
        lines.append(f"Total LLM calls: {total_calls}")
        return '\n'.join(lines)
