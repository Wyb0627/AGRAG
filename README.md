# GORAG
The official code repository of the paper "Graph-based Retrieval Augmented Generation for Dynamic Few-shot Text Classification"

Install all requirements via:

```
pip install -r requirements.txt
```

Then run experiments on the WOS dataset via: 

```
  python run.py --gpu 0 --graphrag --context LLM --LLM llama3 --steiner_tree --edge_weighting tfidf --desc_keywords --shot 1 --online_index all --round 1
```

And run experiments on the Reuters dataset via:

```
  python run.py --gpu 0 --graphrag --context LLM --LLM llama3 --steiner_tree --edge_weighting tfidf --desc_keywords --dataset reuters --no_label_name --shot 1 --online_index all
```

--gpu: The GPU number used;

--dataset: The dataset experimented on;

--no_label_name: Set for Reuters, where the label names are not available;

--LLM: The LLM for use, available LLMs: llama3, llama3.1, qwen2, qwen2.5, mistral;

--edge_weighting: Whether to apply tfidf based edge weighting mechanism or unit weight;

--shot: The number of shots;

--round: The number of dataset split rounds;

--online_index: Whether to apply the online indexing mechanism.

