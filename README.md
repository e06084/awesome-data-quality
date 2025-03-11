# Awesome Data Quality [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of awesome resources, tools, papers, and projects related to data quality across various data types. This repository aims to be a comprehensive reference for researchers and practitioners working with data quality in different domains.

## Contents

- [Introduction](#introduction)
- [Traditional Data](#traditional-data)
  - [Papers](#traditional-papers)
  - [Tools & Projects](#traditional-tools)
- [Large Language Model Data](#large-language-model-data)
  - [Pretraining Data](#pretraining-data)
    - [Papers](#pretraining-papers)
    - [Tools & Projects](#pretraining-tools)
  - [Fine-tuning Data](#fine-tuning-data)
    - [Papers](#fine-tuning-papers)
    - [Tools & Projects](#fine-tuning-tools)
- [Multimodal Data](#multimodal-data)
  - [Papers](#multimodal-papers)
  - [Tools & Projects](#multimodal-tools)
- [Tabular Data](#tabular-data)
  - [Papers](#tabular-papers)
  - [Tools & Projects](#tabular-tools)
- [Time Series Data](#time-series-data)
  - [Papers](#time-series-papers)
  - [Tools & Projects](#time-series-tools)
- [Graph Data](#graph-data)
  - [Papers](#graph-papers)
  - [Tools & Projects](#graph-tools)
- [Data-Centric AI](#data-centric-ai)
  - [Surveys](#data-centric-surveys)
  - [Data Valuation](#data-valuation)
  - [Data Selection](#data-selection)
  - [Benchmarks](#data-centric-benchmarks)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Data quality is a critical aspect of any data-driven application or research. This repository collects resources related to data quality across different data types, including traditional data, large language model data (both pretraining and fine-tuning), multimodal data, and more.

## Traditional Data

This section covers data quality for traditional structured and unstructured data.

### <a name="traditional-papers"></a>Papers

- [Data Cleaning: Problems and Current Approaches](https://www.researchgate.net/publication/220423285_Data_Cleaning_Problems_and_Current_Approaches) - A comprehensive overview of data cleaning approaches.
- [A Survey on Data Quality: Classifying Poor Data](https://ieeexplore.ieee.org/document/7423672) - A survey on data quality issues and classification.

### <a name="traditional-tools"></a>Tools & Projects

- [Great Expectations](https://github.com/great-expectations/great_expectations) - A Python framework for validating, documenting, and profiling data.
- [Deequ](https://github.com/awslabs/deequ) - A library built on top of Apache Spark for defining "unit tests for data".
- [OpenRefine](https://openrefine.org/) - A powerful tool for working with messy data, cleaning it, and transforming it.

## Large Language Model Data

### Pretraining Data

This section covers data quality for large language model pretraining data.

#### <a name="pretraining-papers"></a>Papers

- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) - A large-scale curated dataset for language model pretraining.
- [Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](https://arxiv.org/abs/2103.12028) - An audit of the quality of web-crawled multilingual datasets.
- [Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus](https://arxiv.org/abs/2104.08758) - Documentation of the C4 dataset.

#### <a name="pretraining-tools"></a>Tools & Projects

- [Dolma](https://github.com/allenai/dolma) - A framework for curating and documenting large language model pretraining data.
- [Text Data Cleaner](https://github.com/ChenghaoMou/text-data-cleaner) - A tool for cleaning text data for language model pretraining.
- [CCNet](https://github.com/facebookresearch/cc_net) - Tools for downloading and filtering CommonCrawl data.
- [Dingo](https://github.com/DataEval/dingo) - A comprehensive data quality evaluation tool supporting multiple data sources, types, and modalities.

### Fine-tuning Data

This section covers data quality for large language model fine-tuning data.

#### <a name="fine-tuning-papers"></a>Papers

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - The RLHF paper from Anthropic.
- [Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP](https://arxiv.org/abs/2112.07295) - A study on the importance of data quality over quantity.
- [Data Quality for Machine Learning Tasks](https://arxiv.org/abs/2108.02711) - A survey on data quality for machine learning.

#### <a name="fine-tuning-tools"></a>Tools & Projects

- [LMSYS Chatbot Arena](https://github.com/lm-sys/FastChat) - A platform for evaluating LLM responses.
- [OpenAssistant](https://github.com/LAION-AI/Open-Assistant) - A project to create high-quality instruction-following data.
- [Argilla](https://github.com/argilla-io/argilla) - An open-source data curation platform for LLMs.

## Multimodal Data

This section covers data quality for multimodal data, including image-text pairs, video, and audio.

### <a name="multimodal-papers"></a>Papers

- [LAION-5B: An open large-scale dataset for training next generation image-text models](https://arxiv.org/abs/2210.08402) - A large-scale dataset of image-text pairs.
- [DataComp: In search of the next generation of multimodal datasets](https://arxiv.org/abs/2304.14108) - A benchmark for evaluating data curation strategies.

### <a name="multimodal-tools"></a>Tools & Projects

- [CLIP-Benchmark](https://github.com/LAION-AI/CLIP-Benchmark) - A benchmark for evaluating CLIP models.
- [img2dataset](https://github.com/rom1504/img2dataset) - A tool for efficiently downloading and processing image-text datasets.

## Tabular Data

This section covers data quality for tabular data.

### <a name="tabular-papers"></a>Papers

- [Automating Data Quality Validation for Dynamic Data Ingestion](https://ieeexplore.ieee.org/document/8731379) - A framework for automating data quality validation.
- [A Survey on Data Quality for Machine Learning in Practice](https://arxiv.org/abs/2103.05251) - A survey on data quality issues in machine learning.

### <a name="tabular-tools"></a>Tools & Projects

- [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) - A tool for generating profile reports from pandas DataFrames.
- [DataProfiler](https://github.com/capitalone/DataProfiler) - A Python library for data profiling and data quality validation.

## Time Series Data

This section covers data quality for time series data.

### <a name="time-series-papers"></a>Papers

- [Cleaning Time Series Data: Current Status, Challenges, and Opportunities](https://arxiv.org/abs/2201.05562) - A survey on cleaning time series data.
- [Time Series Data Augmentation for Deep Learning: A Survey](https://arxiv.org/abs/2002.12478) - A survey on time series data augmentation.

### <a name="time-series-tools"></a>Tools & Projects

- [Darts](https://github.com/unit8co/darts) - A Python library for time series forecasting and anomaly detection.
- [tslearn](https://github.com/tslearn-team/tslearn) - A machine learning toolkit dedicated to time series data.

## Graph Data

This section covers data quality for graph data.

### <a name="graph-papers"></a>Papers

- [A Survey on Graph Cleaning Methods for Noise and Errors in Graph Data](https://arxiv.org/abs/2201.00443) - A survey on graph cleaning methods.
- [Graph Data Quality: A Survey from the Database Perspective](https://arxiv.org/abs/2201.05236) - A survey on graph data quality from a database perspective.

### <a name="graph-tools"></a>Tools & Projects

- [DGL](https://github.com/dmlc/dgl) - A Python package for deep learning on graphs.
- [NetworkX](https://github.com/networkx/networkx) - A Python package for the creation, manipulation, and study of complex networks.

## Data-Centric AI

This section focuses on data quality management for machine learning models, following the Data-Centric AI paradigm. It includes papers and resources related to data valuation, data selection, and benchmarks for evaluating data quality in ML pipelines.

### <a name="data-centric-surveys"></a>Surveys

- [Data Quality Awareness: A Journey from Traditional Data Management to Data Science Systems](https://arxiv.org/pdf/2411.03007) - A comprehensive survey on data quality awareness across traditional data management and modern data science systems.
- [A Survey on Data Selection for Language Models](https://arxiv.org/pdf/2402.16827.pdf) - A survey focusing on data selection techniques for language models.
- [Advances, challenges and opportunities in creating data for trustworthy AI](https://www.nature.com/articles/s42256-022-00516-1) - A Nature Machine Intelligence paper discussing the challenges and opportunities in creating high-quality data for AI.
- [Data-centric Artificial Intelligence: A Survey](https://arxiv.org/pdf/2303.10158.pdf) - A comprehensive survey on data-centric AI approaches.
- [Data Management For Large Language Models: A Survey](https://arxiv.org/pdf/2312.01700.pdf) - A survey on data management techniques for large language models.
- [Training Data Influence Analysis and Estimation: A Survey](https://arxiv.org/pdf/2212.04612.pdf) - A survey on methods for analyzing and estimating the influence of training data on model performance.
- [Data Management for Machine Learning: A Survey](https://luoyuyu.vip/files/DM4ML%5FSurvey.pdf) - A TKDE survey on data management techniques for machine learning.
- [Data Valuation in Machine Learning: "Ingredients", Strategies, and Open Challenges](https://www.ijcai.org/proceedings/2022/0782.pdf) - An IJCAI paper on data valuation methods in machine learning.
- [Explanation-Based Human Debugging of NLP Models: A Survey](https://aclanthology.org/2021.tacl-1.90.pdf) - A TACL survey on explanation-based debugging of NLP models.

### <a name="data-valuation"></a>Data Valuation

- [Data Shapley: Equitable Valuation of Data for Machine Learning](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf) - An ICML paper introducing the Data Shapley method for valuing training data.
- [Efficient task-specific data valuation for nearest neighbor algorithms](https://vldb.org/pvldb/vol12/p1610-jia.pdf) - A VLDB paper on efficient data valuation for nearest neighbor algorithms.
- [Towards Efficient Data Valuation Based on the Shapley Value](https://proceedings.mlr.press/v89/jia19a/jia19a.pdf) - An AISTATS paper on efficient data valuation using Shapley values.
- [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/pdf/1703.04730.pdf) - An ICML paper introducing influence functions for understanding model predictions.
- [Data Cleansing for Models Trained with SGD](https://proceedings.neurips.cc/paper%5Ffiles/paper/2019/file/5f14615696649541a025d3d0f8e0447f-Paper.pdf) - A NeurIPS paper on data cleansing for SGD-trained models.

### <a name="data-selection"></a>Data Selection

- [Modyn: Data-Centric Machine Learning Pipeline Orchestration](https://arxiv.org/pdf/2312.06254) - A SIGMOD paper on pipeline orchestration for data-centric machine learning.
- [Data Selection via Optimal Control for Language Models](https://openreview.net/pdf?id=dhAL5fy8wS) - An ICLR paper on optimal control methods for data selection in language models.
- [ADAM Optimization with Adaptive Batch Selection](https://openreview.net/pdf?id=BZrSCv2SBq) - An ICLR paper on adaptive batch selection for ADAM optimization.
- [Adaptive Data Optimization: Dynamic Sample Selection with Scaling Laws](https://openreview.net/pdf?id=aqok1UX7Z1) - An ICLR paper on dynamic sample selection using scaling laws.
- [Selection via proxy: Efficient data selection for deep learning](https://openreview.net/pdf?id=HJg2b0VYDr) - An ICLR paper on efficient data selection using proxy models.

### <a name="data-centric-benchmarks"></a>Benchmarks

- [DataPerf: Benchmarks for Data-Centric AI Development](https://openreview.net/pdf?id=LaFKTgrZMG) - A NeurIPS paper introducing benchmarks for data-centric AI development.
- [OpenDataVal: a Unified Benchmark for Data Valuation](https://openreview.net/pdf?id=eEK99egXeB) - A NeurIPS paper on a unified benchmark for data valuation.
- [Improving multimodal datasets with image captioning](https://proceedings.neurips.cc/paper%5Ffiles/paper/2023/file/45e604a3e33d10fba508e755faa72345-Paper-Datasets%5Fand%5FBenchmarks.pdf) - A NeurIPS paper on improving multimodal datasets with image captioning.
- [Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias](https://proceedings.neurips.cc/paper%5Ffiles/paper/2023/file/ae9500c4f5607caf2eff033c67daa9d7-Paper-Datasets%5Fand%5FBenchmarks.pdf) - A NeurIPS paper on using LLMs as training data generators.
- [dcbench: A Benchmark for Data-Centric AI Systems](https://dl.acm.org/doi/pdf/10.1145/3533028.3533310) - A DEEM paper introducing a benchmark for data-centric AI systems.

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

This repository is licensed under the [MIT License](LICENSE).