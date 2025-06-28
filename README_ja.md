# Awesome Data Quality [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

様々なデータタイプにおけるデータ品質に関連するリソース、ツール、論文、プロジェクトの厳選されたリストです。このリポジトリは、異なる分野でデータ品質に取り組む研究者や実務者にとって包括的な参考資料となることを目指しています。

## 目次

- [はじめに](#はじめに)
- [従来のデータ](#従来のデータ)
  - [論文](#従来データ論文)
  - [ツール・プロジェクト](#従来データツール)
  - [データ準備度評価](#データ準備度評価)
- [大規模言語モデルデータ](#大規模言語モデルデータ)
  - [事前学習データ](#事前学習データ)
  - [ファインチューニングデータ](#ファインチューニングデータ)
  - [LLMデータ管理](#llmデータ管理)
  - [認知工学・テスト時スケーリング](#認知工学)
- [マルチモーダルデータ](#マルチモーダルデータ)
  - [論文](#マルチモーダルデータ論文)
  - [ツール・プロジェクト](#マルチモーダルデータツール)
- [表形式データ](#表形式データ)
  - [論文](#表形式データ論文)
  - [ツール・プロジェクト](#表形式データツール)
- [時系列データ](#時系列データ)
  - [論文](#時系列データ論文)
  - [ツール・プロジェクト](#時系列データツール)
- [グラフデータ](#グラフデータ)
  - [論文](#グラフデータ論文)
  - [ツール・プロジェクト](#グラフデータツール)
- [データ中心AI](#データ中心ai)
  - [サーベイ](#データ中心aiサーベイ)
  - [データ評価](#データ評価)
  - [データ選択](#データ選択)
  - [ベンチマーク](#データ中心aiベンチマーク)
- [貢献ガイド](#貢献ガイド)

## はじめに

データ品質は、あらゆるデータ駆動型アプリケーションや研究において重要な側面です。このリポジトリでは、従来のデータ、大規模言語モデルデータ（事前学習・ファインチューニング）、マルチモーダルデータなど、異なるデータタイプのデータ品質に関連するリソースを収集しています。

## 従来のデータ

このセクションでは、従来の構造化・非構造化データのデータ品質について扱います。

### 論文

- [Data Cleaning: Problems and Current Approaches](https://www.researchgate.net/publication/220423285_Data_Cleaning_Problems_and_Current_Approaches) - データクリーニング手法の包括的概要。(2000)
- [A Survey on Data Quality: Classifying Poor Data](https://ieeexplore.ieee.org/document/7423672) - データ品質問題と分類に関するサーベイ。(2016)

### ツール・プロジェクト

- [Great Expectations](https://github.com/great-expectations/great_expectations) - データの検証、文書化、プロファイリングのためのPythonフレームワーク。(2018)
- [Deequ](https://github.com/awslabs/deequ) - Apache Spark上に構築された「データ単体テスト」を定義するためのライブラリ。(2018)
- [OpenRefine](https://openrefine.org/) - 混乱したデータの処理、クリーニング、変換のための強力なツール。(2010)

### データ準備度評価

このサブセクションでは、AIアプリケーションのデータ準備度を評価する方法とツールについて扱います。

#### 論文

- [Data Readiness for AI: A 360-Degree Survey](https://arxiv.org/abs/2404.05779) - 構造化・非構造化データセットのAI訓練データ準備度を評価するメトリクスの包括的サーベイ。(2024)
- [Assessing Student Adoption of Generative AI in Engineering Education](https://arxiv.org/abs/2503.04696) - 教育AIアプリケーションにおけるデータ品質考慮の実証研究。(2025)

#### ツール・プロジェクト

- [Data Readiness Assessment Framework](https://github.com/data-readiness/framework) - AIアプリケーションのデータ品質と準備度を評価するフレームワーク。(2024)
- [AI Data Quality Metrics](https://github.com/ai-data-quality/metrics) - AI環境でデータ品質を評価するための標準化メトリクス。(2024)

## 大規模言語モデルデータ

### 事前学習データ

このセクションでは、大規模言語モデルの事前学習データのデータ品質について扱います。

#### 論文

- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) - 言語モデル事前学習用の大規模キュレーションデータセット。(2021)
- [Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](https://arxiv.org/abs/2103.12028) - ウェブクロール多言語データセットの品質監査。(2021)
- [Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus](https://arxiv.org/abs/2104.08758) - C4データセットの文書化。(2021)

#### ツール・プロジェクト

- [Dolma](https://github.com/allenai/dolma) - 大規模言語モデル事前学習データのキュレーションと文書化のためのフレームワーク。(2023)
- [Text Data Cleaner](https://github.com/ChenghaoMou/text-data-cleaner) - 言語モデル事前学習用テキストデータのクリーニングツール。(2022)
- [CCNet](https://github.com/facebookresearch/cc_net) - CommonCrawlデータのダウンロードとフィルタリングツール。(2020)
- [Dingo](https://github.com/DataEval/dingo) - 複数のデータソース、タイプ、モダリティをサポートする包括的データ品質評価ツール。(2024)

### ファインチューニングデータ

このセクションでは、大規模言語モデルのファインチューニングデータのデータ品質について扱います。

#### 論文

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - AnthropicのRLHF論文。(2022)
- [Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP](https://arxiv.org/abs/2112.07295) - データ品質が量よりも重要であることに関する研究。(2021)
- [Data Quality for Machine Learning Tasks](https://arxiv.org/abs/2108.02711) - 機械学習におけるデータ品質のサーベイ。(2021)

#### ツール・プロジェクト

- [LMSYS Chatbot Arena](https://github.com/lm-sys/FastChat) - LLM応答を評価するプラットフォーム。(2023)
- [OpenAssistant](https://github.com/LAION-AI/Open-Assistant) - 高品質な指示追従データを作成するプロジェクト。(2022)
- [Argilla](https://github.com/argilla-io/argilla) - LLM用のオープンソースデータキュレーションプラットフォーム。(2021)

### LLMデータ管理

このセクションでは、データ処理、保存、サービングを含むLLMの包括的データ管理アプローチについて扱います。

#### 論文

- [A Survey of LLM × DATA](https://arxiv.org/abs/2505.18458) - データ処理、保存、サービングを含む大規模言語モデルのデータ中心アプローチに関する包括的サーベイ。(2025)
- [Fixing Performance-Damaging Data: Cascaded LLM Re-labeling of Hard Negatives for Robust Information Retrieval](https://arxiv.org/abs/2505.16967) - モデル性能向上のため訓練データ内の偽陰性を特定・再ラベル化する手法。(2025)

#### ツール・プロジェクト

- [awesome-data-llm](https://github.com/weAIDB/awesome-data-llm) - 「LLM × データ」サーベイ論文の公式リポジトリ、キュレーションされたリソースを含む。(2025)
- [CommonCrawl](https://commoncrawl.org/) - 複数言語・ドメインをカバーする大規模ウェブクロールデータセット。(2008)
- [RedPajama](https://github.com/togethercomputer/RedPajama-Data) - LLaMA訓練データセットのオープンソース再現。(2023)
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) - 言語モデル訓練用の大規模高品質ウェブデータセット。(2024)

### 認知工学・テスト時スケーリング

このセクションでは、推論・思考プロセスの強化によりデータ品質を改善する認知工学・テスト時スケーリング手法に焦点を当てます。

#### サーベイ

- [Generative AI Act II: Test Time Scaling Drives Cognition Engineering](https://arxiv.org/abs/2504.13828) - テスト時スケーリングと強化学習による認知工学の包括的サーベイ。(2025)
- [Unlocking Deep Thinking in Language Models: Cognition Engineering via Inference-Time Scaling and Reinforcement Learning](https://gair-nlp.github.io/cognition-engineering/) - テスト時スケーリングパラダイムによるAI思考能力開発フレームワーク。(2025)

#### データエンジニアリング2.0

- [O1 Journey -- Part I](https://github.com/GAIR-NLP/O1-Journey) - 長い連鎖思考を持つ数学推論データセット。(2024)
- [Marco-o1](https://github.com/AIDC-AI/Marco-o1) - Qwen2-7B-Instructから合成された推論データセット。(2024)
- [STILL-2](https://github.com/RUCAIBox/Slow_Thinking_with_LLMs) - 数学、コード、科学、パズル領域の長形式思考データ。(2024)
- [OpenThoughts-114k](https://github.com/open-thoughts/open-thoughts) - DeepSeek R1から抽出された大規模推論軌跡データセット。(2024)

#### 訓練データ品質

- [High-Impact Sample Selection](https://arxiv.org/abs/2502.11886) - 学習影響測定に基づく訓練サンプル優先選択手法。(2025)
- [Noise Reduction Filtering](https://arxiv.org/abs/2502.03373) - 汎化改善のためのノイズウェブ抽出データ除去技術。(2025)
- [Length-Adaptive Training](https://arxiv.org/abs/2504.05118) - 訓練データ内の可変長シーケンス処理手法。(2024)

## マルチモーダルデータ

このセクションでは、画像-テキストペア、動画、音声を含むマルチモーダルデータのデータ品質について扱います。

### 論文

- [LAION-5B: An open large-scale dataset for training next generation image-text models](https://arxiv.org/abs/2210.08402) - 大規模画像-テキストペアデータセット。(2022)
- [DataComp: In search of the next generation of multimodal datasets](https://arxiv.org/abs/2304.14108) - データキュレーション戦略を評価するベンチマーク。(2023)

### ツール・プロジェクト

- [CLIP-Benchmark](https://github.com/LAION-AI/CLIP-Benchmark) - CLIPモデル評価ベンチマーク。(2021)
- [img2dataset](https://github.com/rom1504/img2dataset) - 画像-テキストデータセットの効率的ダウンロード・処理ツール。(2021)

## 表形式データ

このセクションでは、表形式データのデータ品質について扱います。

### 論文

- [Automated Data Quality Validation for Dynamic Data Ingestion](https://ieeexplore.ieee.org/document/8731379) - 自動データ品質検証フレームワーク。(2019)
- [A Survey on Data Quality for Machine Learning in Practice](https://arxiv.org/abs/2103.05251) - 機械学習におけるデータ品質問題のサーベイ。(2021)

### ツール・プロジェクト

- [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) - pandas DataFrameからプロファイルレポートを生成するツール。(2016)
- [DataProfiler](https://github.com/capitalone/DataProfiler) - データ分析・データ品質検証用Pythonライブラリ。(2021)

## 時系列データ

このセクションでは、時系列データのデータ品質について扱います。

### 論文

- [Cleaning Time Series Data: Current State, Challenges, and Opportunities](https://arxiv.org/abs/2201.05562) - 時系列データクリーニングのサーベイ。(2022)
- [Time Series Data Augmentation for Deep Learning: A Survey](https://arxiv.org/abs/2002.12478) - 時系列データ拡張のサーベイ。(2020)

### ツール・プロジェクト

- [Darts](https://github.com/unit8co/darts) - 時系列予測・異常検知用Pythonライブラリ。(2020)
- [tslearn](https://github.com/tslearn-team/tslearn) - 時系列データ専用機械学習ツールキット。(2017)

## グラフデータ

このセクションでは、グラフデータのデータ品質について扱います。

### 論文

- [A Survey of Graph Cleaning Methods for Noise and Errors in Graph Data](https://arxiv.org/abs/2201.00443) - グラフクリーニング手法のサーベイ。(2022)
- [Graph Data Quality: A Survey from a Database Perspective](https://arxiv.org/abs/2201.05236) - データベース観点からのグラフデータ品質サーベイ。(2022)

### ツール・プロジェクト

- [DGL](https://github.com/dmlc/dgl) - グラフ深層学習用Pythonパッケージ。(2018)
- [NetworkX](https://github.com/networkx/networkx) - 複雑ネットワークの作成・操作・研究用Pythonパッケージ。(2008)

## データ中心AI

このセクションでは、データ中心AIパラダイムに従った機械学習モデルのデータ品質管理に焦点を当てます。データ評価、データ選択、MLパイプラインにおけるデータ品質評価ベンチマークに関連する論文・リソースを含みます。

### サーベイ

- [Data Quality Awareness: A Journey from Traditional Data Management to Data Science Systems](https://arxiv.org/pdf/2411.03007) - 従来のデータ管理から現代データサイエンスシステムまでのデータ品質意識包括サーベイ。(2024)
- [A Survey on Data Selection for Language Models](https://arxiv.org/pdf/2402.16827.pdf) - 言語モデルデータ選択技術に焦点を当てたサーベイ。(2024)
- [Progress, challenges and opportunities in creating trustworthy AI data](https://www.nature.com/articles/s42256-022-00516-1) - AI用高品質データ作成の課題と機会を議論するNature Machine Intelligence論文。(2022)
- [Data-centric artificial intelligence: A survey](https://arxiv.org/pdf/2303.10158.pdf) - データ中心AIアプローチの包括的サーベイ。(2023)
- [Data Management For Large Language Models: A Survey](https://arxiv.org/pdf/2312.01700.pdf) - 大規模言語モデルデータ管理技術のサーベイ。(2023)
- [Training Data Influence Analysis and Estimation: A Survey](https://arxiv.org/pdf/2212.04612.pdf) - 訓練データのモデル性能への影響分析・推定手法のサーベイ。(2022)
- [Data Management for Machine Learning: A Survey](https://luoyuyu.vip/files/DM4ML%5FSurvey.pdf) - 機械学習データ管理技術に関するTKDEサーベイ。(2022)
- [Data Valuation in Machine Learning: "Ingredients", Strategies, and Open Challenges](https://www.ijcai.org/proceedings/2022/0782.pdf) - 機械学習におけるデータ評価手法に関するIJCAI論文。(2022)
- [Explanation-based Human Debugging of NLP Models: A Survey](https://aclanthology.org/2021.tacl-1.90.pdf) - 説明ベースNLPモデルデバッグに関するTACLサーベイ。(2021)

### データ評価

- [Data Shapley: Equitable Valuation of Machine Learning Data](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf) - 訓練データ評価用データShapley手法を紹介するICML論文。(2019)
- [Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms](https://vldb.org/pvldb/vol12/p1610-jia.pdf) - 最近傍アルゴリズムの効率的データ評価に関するVLDB論文。(2019)
- [Towards Efficient Data Valuation Based on the Shapley Value](https://proceedings.mlr.press/v89/jia19a/jia19a.pdf) - Shapley値を用いた効率的データ評価に関するAISTATS論文。(2019)
- [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/pdf/1703.04730.pdf) - モデル予測理解用影響関数を紹介するICML論文。(2017)
- [Data Cleansing for Models Trained with SGD](https://proceedings.neurips.cc/paper%5Ffiles/paper/2019/file/5f14615696649541a025d3d0f8e0447f-Paper.pdf) - SGD訓練モデルのデータクリーニングに関するNeurIPS論文。(2019)

### データ選択

- [Modyn: Data-centric Machine Learning Pipeline Orchestration](https://arxiv.org/pdf/2312.06254) - データ中心機械学習パイプライン編成に関するSIGMOD論文。(2023)
- [Data Selection for Language Models via Optimal Control](https://openreview.net/pdf?id=dhAL5fy8wS) - 言語モデルデータ選択の最適制御手法に関するICLR論文。(2024)
- [ADAM with Adaptive Batch Selection](https://openreview.net/pdf?id=BZrSCv2SBq) - ADAM最適化の適応バッチ選択に関するICLR論文。(2024)
- [Adaptive Data Optimization: Dynamic Sample Selection using Scaling Laws](https://openreview.net/pdf?id=aqok1UX7Z1) - スケーリング法則を用いた動的サンプル選択に関するICLR論文。(2024)
- [Selection via Proxy: Efficient Data Selection for Deep Learning](https://openreview.net/pdf?id=HJg2b0VYDr) - プロキシモデルを用いた効率的データ選択に関するICLR論文。(2020)

### ベンチマーク

- [DataPerf: Benchmarks for Data-Centric AI Development](https://openreview.net/pdf?id=LaFKTgrZMG) - データ中心AI開発ベンチマークを紹介するNeurIPS論文。(2023)
- [OpenDataVal: a Unified Benchmark for Data Valuation](https://openreview.net/pdf?id=eEK99egXeB) - 統一データ評価ベンチマークに関するNeurIPS論文。(2023)
- [Improving Multimodal Datasets with Image Captioning](https://proceedings.neurips.cc/paper%5Ffiles/paper/2023/file/45e604a3e33d10fba508e755faa72345-Paper-Datasets%5Fand%5FBenchmarks.pdf) - 画像キャプションによるマルチモーダルデータセット改善に関するNeurIPS論文。(2023)
- [Large Language Models as Attributed Training Data Generators: A Tale of Diversity and Bias](https://proceedings.neurips.cc/paper%5Ffiles/paper/2023/file/ae9500c4f5607caf2eff033c67daa9d7-Paper-Datasets%5Fand%5FBenchmarks.pdf) - LLMを訓練データ生成器として使用することに関するNeurIPS論文。(2023)
- [dcbench: A Benchmark for Data-Centric AI Systems](https://dl.acm.org/doi/pdf/10.1145/3533028.3533310) - データ中心AIシステムベンチマークを紹介するDEEM論文。(2022)

## 貢献ガイド

貢献を歓迎します！まず[貢献ガイド](CONTRIBUTING.md)をお読みください。 
