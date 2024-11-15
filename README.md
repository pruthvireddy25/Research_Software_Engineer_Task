# Research Software Engineer Task
This project provides a streamlined solution for identifying and categorizing academic papers that apply deep learning methods in the fields of virology and epidemiology. With a dataset of 11,450 papers, this tool leverages advanced NLP techniques to semantically filter relevant research papers and classify them by method type and specific methods used.

## Project Overview
The goal of this solution is to:
1.	Filter and Select papers specifically focused on deep learning methods applied in virology and epidemiology, discarding irrelevant papers.
2.	Classify relevant papers based on the type of deep learning method applied (e.g., text mining, computer vision).
3.	Extract Method Names: Identify specific deep learning methods used (e.g., "CNN", "RNN", "LSTM") within the relevant papers for detailed analysis.

This approach provides a structured, efficient way to manage and analyze large academic datasets.

## Approach
**Semantic Filtering Using NLP Techniques:**

**Technique:** Sentence embeddings were created using SentenceTransformers (paraphrase-MiniLM-L6-v2). The embeddings provide a dense, context-aware representation of both the query and paper abstracts, allowing for precise filtering based on semantic content.

**Why this is Effective:**
- Unlike traditional keyword-based filtering, semantic filtering detects relevance even when authors use different terminologies, synonyms, or phrasings.
- This approach minimizes false positives (irrelevant papers containing keywords by coincidence) and false negatives (relevant papers using non-standard terms).
- By setting an appropriate similarity threshold (0.3-0.5), the system effectively captures nuanced semantic meaning, ensuring a high-quality subset of truly relevant research.

**Classification and Method Extraction:**
- **Classification:** The classification system analyzes each abstract to determine if the paper focuses on Text Mining, Computer Vision, Both, or Other. This structure supports deeper analysis by organizing research around core technique types.
- **Method Extraction Using Sentence Similarity:**
  - Each abstract is semantically compared with method-related keywords (e.g., "convolutional neural network", "GAN"). This step identifies the deep learning methods even if phrased differently.
  - Sentence similarity here is crucial because it captures the essence of techniques without requiring exact phrase matching, accommodating diverse terminologies and improving extraction precision.

## Resulting Dataset Statistics
- Initial Dataset: 11,450 records from a keyword-based PubMed search, with metadata and abstracts.
- After Semantic Filtering: 1382 papers met the relevance threshold of 0.4.
- Classification Results: For Method type:
  - text mining – 84 papers
  - computer vision – 56 papers
  - both – 83 papers
  - other – 1159 papers

  For specific Method name extraction:
  - recurrent neural network – 280 papers
  - convolutional neural network – 143 papers
  - RNN – 106 papers
  - GAN – 38 papers
  - CNN – 32 papers
  - LSTM – 19 papers
  - Autoencoder – 13 papers
  - Not specified – 751 papers

## Instructions to Run:
- Prior to running, install required dependencies by executing: `pip install sentence-transformers`.
- Execute task_file.py to process the dataset and output the filtered, classified, and extracted data into filtered_virology_papers.csv.
- Filtered_virology_papers.csv output file includes columns such as paper ID, title, abstract, is relevant, method type, and method name.

## Conclusion
This solution provides a robust, NLP-driven tool for filtering, classifying, and analyzing deep learning applications in virology and epidemiology research. By using semantic similarity, it surpasses simple keyword-based filters, offering researchers a refined and insightful dataset.
