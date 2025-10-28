# AGRAG
The official code repository of the paper "AGRAG: Advanced Graph-based Retrieval-Augmented Generation for LLMs"

## GraphRAG-bench

To run GraphRAG-bench: 

1. Substitute the run_hipporag2.py of the original [GraphRAG-bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main) repo with the provided on in /GraphRAG_bench folder.
2. Download [HippoRAG2](https://github.com/ianliuwd/HippoRAG2/tree/main)'s code, and copy everything in the src folder into the folder where run_hipporag2.py is (We use the HippoRAG package version 2.0.0a3).
3. Substitute the respective files within the src folder with the provided one in /GraphRAG_bench folder.

Then run AGRAG on the novel or medical dataset via: 
```
  python run_hipporag2.py --include_passage_nodes --subset novel --retrieval_mode hybrid --ner_threshold 0.5 --llm_api_key YOUR_API_KEY_IF_NEEDED
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

--gpu: The GPU number used;

--dataset: The dataset experimented on;

--no_label_name: Set for Reuters, where the label names are not available;

--LLM: The LLM for use, available LLMs: llama3, llama3.1, qwen2, qwen2.5, mistral;

--edge_weighting: Whether to apply tfidf based edge weighting mechanism or unit weight;

--shot: The number of shots;

--round: The number of dataset split rounds set to 4 for full experiment running;

--online_index: Whether to apply the online indexing mechanism.


### The code to run on GraphRAG-bench is partially based on [GraphRAG-bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main) and [HippoRAG2](https://github.com/ianliuwd/HippoRAG2/tree/main)'s GitHub repo, huge thanks to their contributions!
