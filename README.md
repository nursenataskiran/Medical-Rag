# Medical RAG 

A retrieval-augmented generation (RAG) system for answering medical questions using recent PubMed abstracts.  
The project combines BM25, semantic, and hybrid retrieval methods, then uses Gemini to generate grounded answers based only on retrieved evidence.  
It also includes an evaluation pipeline for comparing retrievers using ranking metrics such as Precision@5, MRR, and nDCG@5.

---

## Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/nursenataskiran/Medical-Rag.git
cd Medical-Rag
```

### 2. Create and Activate a Virtual Environment
- **Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

- **macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Configure Environment Variables
Create a ```.env``` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

### 5. Run the Medical RAG CLI
```
medrag
```
You can then enter medical questions interactively.
Example:
```
Query> What are the latest guidelines for managing type 2 diabetes?
```
Type ```exit``` or ```quit``` to stop the program.

## Data Files

The `data/` directory contains the resources used by the retrieval and evaluation pipeline:

- `medical_terms.csv`  
  Seed medical terms used for PubMed collection.

- `pubmed_articles.json`  
  Retrieved PubMed abstracts used as the document corpus for RAG. Generated from the seed terms in `medical_terms.csv` using the PubMed fetch pipeline.

- `eval/evaluation_pool_blind.csv`  
  Candidate retrieval pool before relevance labeling.

- `eval/evaluation_pool_labeled.csv`  
  Manually labeled relevance judgments used for evaluation metrics.

## Retrieval Strategy

I implemented three retrieval approaches and compared them experimentally: BM25, semantic retrieval, and hybrid retrieval.

---

### BM25 Retriever

BM25 was used as the sparse lexical baseline. It is effective for exact keyword overlap, abbreviations, and terminology-heavy medical queries where token matching remains important.

BM25 ranks documents using term frequency, inverse document frequency, and document length normalization. This is particularly relevant for PubMed abstracts, where some documents are short and highly focused while others are longer review-style summaries. Without length normalization, longer abstracts may receive inflated scores simply because they contain more matching tokens.

Because the evaluation set includes Turkish queries while the PubMed corpus is English, I added a lightweight Turkish-to-English normalization dictionary for common medical phrases (e.g., disease names, treatment terms, diagnostic terms). This avoided introducing external translation dependencies while improving cross-lingual lexical matching.

Given more time or a larger real-world corpus, I would replace this rule-based normalization with a stronger multilingual translation layer or LLM-based query rewriting pipeline.

#### Parameter Selection (`k1`, `b`)

I tested several BM25 configurations and found retrieval quality to be relatively stable across settings on this small, domain-bounded corpus.

For term-frequency saturation (`k1`), I tested:

- `0.8`
- `1.2`
- `1.5`
- `1.8`

These settings produced only minor ranking differences. In practice, changing `k1` mostly affected score magnitudes rather than substantially changing document order.

Since `1.5` provided a good balance while remaining stable across queries, I selected:

- `k1 = 1.5`

For document length normalization (`b`), I tested multiple values and observed the clearest qualitative improvement when lowering:

- `b: 0.75 → 0.5`

This slightly improved ranking behavior for the Turkish celiac disease query by promoting a more relevant biopsy-related article above a less relevant diverticulitis article.

This suggested that the default length penalty was slightly too strong for this compact PubMed abstract corpus.

Final provisional BM25 setting:

- `k1 = 1.5`
- `b = 0.5`

To inspect BM25 retrieval outputs directly, the BM25 script can be run independently. I defined a small set of base queries in the script to quickly observe how lexical retrieval behaves on both English and Turkish phrasing, especially for terminology-heavy questions.
```bash
python -m scripts.run_bm25_ret
```
---

### Semantic Retriever

Semantic retrieval was used as the dense retrieval component to capture meaning beyond exact keyword overlap. This was especially valuable for the project setting, where Turkish-speaking doctors query an English PubMed corpus.

Unlike BM25, semantic search can retrieve relevant documents even when the query wording differs from the article text, making it more effective for multilingual and paraphrased medical questions.

For embeddings, I used intfloat/multilingual-e5-small through sentence-transformers.

This model was selected because it offers:

* Strong multilingual retrieval performance
* Good speed / quality tradeoff
* Much smaller footprint than larger alternatives
* Practical local inference for a take-home project
  
To inspect semantic retrieval outputs directly, the semantic retriever script can be run independently. I defined a small set of base queries in the script to quickly observe how embedding-based retrieval behaves on both English and Turkish phrasing, especially for paraphrased or concept-based medical questions.
```bash
python -m scripts.run_semantic_ret
```
---

### Hybrid Retriever


The hybrid retriever combines BM25 and semantic retrieval using **Reciprocal Rank Fusion (RRF)**, aiming to merge lexical precision with semantic recall.

Instead of combining raw retrieval scores, RRF combines ranked lists using document positions:

```text
score(d) = Σ 1 / (k + rank_i(d))
```

#### RRF Parameter Selection (`k`)

The RRF parameter `k` controls how strongly rank position influences the fused score.

- Smaller values (e.g., `10–20`) place more emphasis on documents that appear near the top of both retrieval lists.
- `k = 60` is a common default and produced stable results.
- Very large values such as `k = 1000` made fused scores nearly identical across candidates, reducing ranking discrimination.

For this small PubMed abstract corpus with relatively shallow ranked lists, `k = 20` provided the best balance between robustness and useful rank separation, so it was selected for the final hybrid retriever.

To inspect hybrid retrieval outputs directly, the hybrid retriever script can be run independently. I defined a small set of base queries in the script to quickly observe how fused retrieval behaves on both English and Turkish phrasing by combining lexical BM25 matching with semantic similarity signals.
```bash
python -m scripts.run_hybrid_ret
```

### Why use rank position instead of raw scores when combining BM25 with cosine similarity?

Rank positions are preferred because **BM25 and cosine similarity scores are not directly comparable**.

- **BM25 scores** depend on corpus statistics such as term frequency, inverse document frequency (IDF), and document length normalization. Their scale can vary significantly across queries and corpora.
- **Cosine similarity scores** are bounded vector similarity values (typically between -1 and 1, or 0 and 1 depending on implementation).

Because these scoring systems have different ranges and meanings, directly adding or averaging raw scores can produce unstable or misleading results.

Using **rank positions** avoids this calibration problem. Each retriever only contributes the relative ordering of documents, making fusion:

- more robust across heterogeneous retrievers  
- less sensitive to score scaling differences  
- easier to generalize across datasets and query types
  
This is why Reciprocal Rank Fusion (RRF) uses document ranks rather than raw retrieval scores.


## Evaluation

I evaluated the three retrieval methods using a small benchmark set designed to reflect realistic medical information needs across both English and Turkish queries.

### Benchmark Queries

Five benchmark queries were selected to cover different retrieval intents.
These queries were intentionally mixed across:

- English and Turkish phrasing
- treatment-oriented questions
- guideline / criteria questions
- factual evidence retrieval

---

### Retrieval Methods Compared

The following systems were evaluated:

- **BM25**
- **Semantic Retrieval** (`multilingual-e5-small`)
- **Hybrid Retrieval** (BM25 + Semantic using Reciprocal Rank Fusion)

Each retriever returned **Top-10** candidates per query. 

---

### Blinded Pooled Relevance Set

To reduce evaluator bias, I used a pooled judgment setup:

1. Top-10 results from all three retrievers were merged into a query-level pool  
2. Duplicate PMIDs within each query were removed  
3. Retrieval-source information was removed before labeling  

This ensured that documents were judged on relevance rather than by knowing which retriever produced them.

---

### Relevance Labels

Each query-document pair received a graded relevance score:

- **3** = perfect match  
- **2** = strongly relevant  
- **1** = weakly relevant  
- **0** = irrelevant 
This allowed the use of ranking-aware evaluation metrics.

---

### Metrics Chosen

#### Primary Metric: nDCG@5

I selected **nDCG@5** as the primary metric because it rewards:

- highly relevant documents appearing near the top
- graded relevance labels
- ranking quality rather than simple hit counting

#### Secondary Metrics

- **MRR** — measures how early the first strong result appears  
- **Precision@5** — measures relevance density in the top 5 results

#### Why Recall Was Omitted

Recall was intentionally not reported because exhaustive corpus-wide relevance labeling was not available. 

---

### Final Results

| Method | Precision@5 | MRR | nDCG@5 |
|--------|-------------|-----|--------|
| Semantic | 0.84 | 1.00 | 0.8357 |
| Hybrid (RRF) | 0.76 | 1.00 | 0.8083 |
| BM25 | 0.72 | 0.90 | 0.7421 |

---

### Interpretation

Semantic retrieval achieved the best average scores across all three metrics. Likely reasons include:

- strong multilingual Turkish → English matching  
- small corpus size  
- better semantic intent understanding

These results suggest that dense retrieval was well suited to this compact benchmark corpus.

A key takeaway from this experiment is that the best offline scorer is not always the best production choice. Final retriever selection depends not only on mean metrics, but also on consistency, failure modes, and robustness across query styles.

---

### Practical Decision

Although semantic retrieval won on average metrics, query-level testing showed higher variance across the benchmark set. It performed extremely well on some queries, but underperformed on several treatment.

The hybrid retriever had slightly lower mean scores, but produced more consistent rankings with fewer severe misses across mixed query types.

Because a medical QA system should prioritize reliability over small gains in average performance, I selected the **Hybrid Retriever** for the final RAG pipeline.

To reproduce the evaluation results, run the evaluator script. It computes retrieval metrics across the five predefined benchmark queries and prints the aggregated scores for each retriever.

```
python -m src.evaluation.evaluate
```

## Generation Strategy

For the answer generation stage, I used **Gemini 2.5 Flash** via the Google AI Studio

This model was selected because it provides a strong balance of:

- fast response time for interactive CLI usage  
- low-cost / accessible API usage  
- solid instruction-following quality  
- reliable multilingual output, especially Turkish responses over English source material

This was particularly relevant for the project setting, where Turkish-speaking doctors query an English PubMed corpus.

To reduce hallucination risk, the generation step was constrained through a strict system prompt:

- answer **only** from retrieved context  
- clearly state when evidence is insufficient  
- always answer in Turkish  
- preserve core medical terminology in original form when appropriate  
- cite sources using PMID or article title 

The final pipeline retrieves top documents using the selected retriever, builds a structured context block, and sends that context to Gemini for grounded answer generation.
### Demo Answers
<img width="1417" height="387" alt="2" src="https://github.com/user-attachments/assets/c9c5ae34-b1dc-49e6-8d4e-84c0677a1b56" />
<br>
<img width="1415" height="592" alt="1" src="https://github.com/user-attachments/assets/53c2c505-26d9-4b7d-b213-d668ceb0194d" />

## Hardest Problem

The hardest part of the project was identifying whether poor answers were caused by retrieval quality or corpus limitations.

Initially, offline metrics suggested that the semantic retriever was the strongest option. However, after integrating the full RAG pipeline, I observed inconsistent performance on some clinically important queries such as guideline, dosing, and treatment questions.

I first treated this as a retrieval problem and experimented with the Hybrid Retriever for better robustness. This improved consistency, but some queries still failed to produce useful answers.

For example:

> What are the latest guidelines for managing type 2 diabetes?

Even after retrieval improvements, the returned documents were diabetes-related but did not contain actual guideline or standards-of-care sources.

This led me to manually inspect the dataset itself. I found that the main issue was not ranking quality, but **corpus coverage**: the collected PubMed subset simply contained too few directly relevant papers for certain query types.

In other words, the retriever was already performing close to its practical ceiling given the available corpus.

A likely next step would have been to expand the corpus by retrieving more articles per seed term, broadening query terms, or adding dedicated guideline-oriented sources. However, I intentionally kept the dataset size limited in order to focus on retrieval behavior under a constrained corpus setting.

The main lesson was that in RAG systems, poor answers are often caused by data limitations rather than retrieval algorithms alone.

