# AGRAG
The official code repository of the paper "AGRAG: Advanced Graph-based Retrieval-Augmented Generation for LLMs"

## GraphRAG-bench

To run GraphRAG-bench: 

1. Substitute the run_hipporag2.py of the original [GraphRAG-bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main) repo with the provided on in /GraphRAG_bench folder.
2. Download [HippoRAG2](https://github.com/ianliuwd/HippoRAG2/tree/main)'s code, and copy everything in the src folder into the folder where run_hipporag2.py is (We use the HippoRAG package version 2.0.0a3).
3. Substitute the respective files within the src folder with those provided in ./GraphRAG_bench/HippoRAG2 folder.
4. Set up VLLM services.

Then run AGRAG on the novel or medical dataset via: 
```
  python run_hipporag2.py --include_passage_nodes --subset novel --retrieval_mode hybrid --ner_threshold 0.5 --llm_base_url VLLM_SERVICE_URL
```

To evaluate with VLLM, you may use the generation_eval_vllm.py file provided in the /GraphRAG_bench folder. The usage is the same as generation_eval.py of GraphRAG-bench.


## WOS and IFS-REL


Install all requirements via:

```
  pip install -r requirements.txt
```

Then run experiments on the WOS dataset via: 

```
  python run.py --gpu 0 --graphrag --context LLM --LLM llama3 --steiner_tree --edge_weighting tfidf --desc_keywords --shot 1 --online_index all --round 4
```

And run experiments on the IFS-REL dataset via:

```
  python run.py --gpu 0 --graphrag --context LLM --LLM llama3 --steiner_tree --edge_weighting tfidf --desc_keywords --dataset reuters --no_label_name --shot 1 --online_index all
```

## Parameters


--gpu: The GPU number used;

--dataset: The dataset experimented on;

--retrieval_mode: Whether to apply dense or hybrid retrieval;

--ner_threshold: The entity extraction threshold;

--max_ngram_length: The maximum n-gram considered when entity extraction;

--include_passage_nodes: Whether to include passage nodes into the graph;

--no_label_name: Set for Reuters, where the label names are not available;

--LLM: The LLM for use, available LLMs;

--edge_weighting: Whether to apply tfidf based edge weighting mechanism or unit weight;

--shot: The number of shots;

--round: The number of dataset split rounds set to 4 for full experiment running;

--online_index: Whether to apply the online indexing mechanism.


### The code to run on GraphRAG-bench is partially based on [GraphRAG-bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main) and [HippoRAG2](https://github.com/ianliuwd/HippoRAG2/tree/main)'s GitHub repo, huge thanks to their contributions!
