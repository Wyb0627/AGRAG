# GERAG
The official code repository of the paper "GERAG: A Graph-based Efficient Retrieval Augmented Generation Framework"

Install all requirements via:

```
pip install -r requirements.txt
```

Set your huggingface token at line 38 of the run.py in case any downloading is needed, you can comment out line 38 if LLMs are already downloaded. 

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


The code to run on GraphRAG-bench is partially based on [GraphRAG-bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main) and [HippoRAG2](https://github.com/ianliuwd/HippoRAG2/tree/main)'s GitHub repo.
