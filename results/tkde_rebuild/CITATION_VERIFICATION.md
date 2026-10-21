# Citation Verification

All **50** entries in `LITERATURE_MATRIX.csv` were checked against a DOI resolver, official proceedings/journal page, PMLR/JMLR/ACL anthology record, or the cited arXiv record. The matrix records publication status so preprints are not presented as peer-reviewed work. No bibliography entry may be inserted unless it maps to one of these verified records or is added with the same verification fields.

## Status cautions

- `weber2019aml` must be described as workshop/preprint.
- `wu2023grande` must be described as preprint.
- `zhou2026wild` must be described as preprint.
- `dehghani2021lottery` must be described as preprint.
- `bordes2025evalfacts` must be described as preprint.
- `oh2026sota` must be described as preprint.

- Elliptic has no verified archival ACM DOI; cite the 2019 arXiv/workshop paper.
- GRANDE has no verified peer-reviewed version in this audit.
- OGB and BenchmarkCards have official proceedings pages but no verified paper DOI in the inspected record.
- Do not reproduce unstable BetterBench checklist counts; title, authors, venue, year, DOI, and conceptual relevance are stable.
- Do not claim that FraudShiftBench is the first dynamic, protocol-aware, resource-aware, or deployment-oriented graph-anomaly benchmark.

## Verified entries

- `weber2019aml` — Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. arXiv:1908.02591; https://arxiv.org/abs/1908.02591 (workshop/preprint).
- `liu2020graphconsis` — Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection. 10.1145/3397271.3401253; https://doi.org/10.1145/3397271.3401253 (peer reviewed).
- `dou2020caregnn` — Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters. 10.1145/3340531.3411903; https://doi.org/10.1145/3340531.3411903 (peer reviewed).
- `liu2021pcgnn` — Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection. 10.1145/3442381.3449989; https://doi.org/10.1145/3442381.3449989 (peer reviewed).
- `tang2022bwgnn` — Rethinking Graph Neural Networks for Anomaly Detection. PMLR 162; https://proceedings.mlr.press/v162/tang22b.html (peer reviewed).
- `huang2022dgraph` — DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection. 10.52202/068431-1654; https://proceedings.neurips.cc/paper_files/paper/2022/hash/8f1918f71972789db39ec0d85bb31110-Abstract-Datasets_and_Benchmarks.html (peer reviewed).
- `tang2023gadbench` — GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection. 10.52202/075280-1289; https://proceedings.neurips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html (peer reviewed).
- `altman2023ibmaml` — Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. 10.52202/075280-1300; https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html (peer reviewed).
- `egressy2024directed` — Provably Powerful Graph Neural Networks for Directed Multigraphs. 10.1609/aaai.v38i10.29069; https://ojs.aaai.org/index.php/AAAI/article/view/29069 (peer reviewed).
- `wu2023grande` — GRANDE: a neural model over directed multigraphs with application to anti-money laundering. arXiv:2302.02101; https://arxiv.org/abs/2302.02101 (preprint).
- `lin2024fraudgt` — FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection. 10.1145/3677052.3698648; https://research.ibm.com/publications/fraudgt-a-simple-effective-and-efficient-graph-transformer-for-financial-fraud-detection (peer reviewed).
- `blanusa2024gfp` — Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection. 10.1145/3677052.3698674; https://authors.library.caltech.edu/records/vav4g-hxz75 (peer reviewed).
- `hua2026bag` — BAG: Benchmarking Anomaly Detection on Dynamic Graphs. 10.1609/aaai.v40i17.38510; https://ojs.aaai.org/index.php/AAAI/article/view/38510 (peer reviewed).
- `zhou2026wild` — GAD in the Wild: Benchmarking Graph Anomaly Detection under Realistic Deployment Challenges. arXiv:2605.07133; https://arxiv.org/abs/2605.07133 (preprint).
- `pourhabibi2020survey` — Fraud Detection: A Systematic Literature Review of Graph-Based Anomaly Detection Approaches. 10.1016/j.dss.2020.113303; https://doi.org/10.1016/j.dss.2020.113303 (peer reviewed).
- `hu2020ogb` — Open Graph Benchmark: Datasets for Machine Learning on Graphs. official proceedings; https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html (peer reviewed).
- `poursafaei2022better` — Towards Better Evaluation for Dynamic Link Prediction. 10.52202/068431-2386; https://papers.neurips.cc/paper_files/paper/2022/hash/d49042a5d49818711c401d34172f9900-Abstract-Datasets_and_Benchmarks.html (peer reviewed).
- `huang2023tgb` — Temporal Graph Benchmark for Machine Learning on Temporal Graphs. 10.52202/075280-0099; https://proceedings.neurips.cc/paper_files/paper/2023/hash/066b98e63313162f6562b35962671288-Abstract-Datasets_and_Benchmarks.html (peer reviewed).
- `gastinger2024tgb2` — TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs. 10.52202/079017-4450; https://proceedings.neurips.cc/paper_files/paper/2024/hash/fda026cf2423a01fcbcf1e1e43ee9a50-Abstract-Datasets_and_Benchmarks_Track.html (peer reviewed).
- `huang2024benchtemp` — BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks. 10.1109/ICDE60146.2024.00310; https://arxiv.org/abs/2308.16385 (peer reviewed).
- `gui2022good` — GOOD: A Graph Out-of-Distribution Benchmark. 10.52202/068431-0150; https://proceedings.neurips.cc/paper_files/paper/2022/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html (peer reviewed).
- `cawley2010selection` — On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR 11; https://jmlr.org/beta/papers/v11/cawley10a.html (peer reviewed).
- `kaufman2012leakage` — Leakage in Data Mining: Formulation, Detection, and Avoidance. 10.1145/2382577.2382579; https://doi.org/10.1145/2382577.2382579 (peer reviewed).
- `kapoor2023leakage` — Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. 10.1016/j.patter.2023.100804; https://doi.org/10.1016/j.patter.2023.100804 (peer reviewed).
- `gorman2019splits` — We Need to Talk about Standard Splits. 10.18653/v1/P19-1267; https://aclanthology.org/P19-1267/ (peer reviewed).
- `bender2018data` — Data Statements for Natural Language Processing. 10.1162/tacl_a_00041; https://doi.org/10.1162/tacl_a_00041 (peer reviewed).
- `mitchell2019modelcards` — Model Cards for Model Reporting. 10.1145/3287560.3287596; https://doi.org/10.1145/3287560.3287596 (peer reviewed).
- `gebru2021datasheets` — Datasheets for Datasets. 10.1145/3458723; https://doi.org/10.1145/3458723 (peer reviewed).
- `pushkarna2022datacards` — Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI. 10.1145/3531146.3533231; https://doi.org/10.1145/3531146.3533231 (peer reviewed).
- `jacobs2021measurement` — Measurement and Fairness. 10.1145/3442188.3445901; https://doi.org/10.1145/3442188.3445901 (peer reviewed).
- `dehghani2021lottery` — The Benchmark Lottery. arXiv:2107.07002; https://arxiv.org/abs/2107.07002 (preprint).
- `reuel2024betterbench` — BetterBench: Assessing AI Benchmarks, Uncovering Issues, and Establishing Best Practices. 10.52202/079017-0685; https://proceedings.neurips.cc/paper_files/paper/2024/hash/26889e8359e7ef8a7f5d77457364ca55-Abstract-Datasets_and_Benchmarks_Track.html (peer reviewed).
- `sokol2025benchmarkcards` — BenchmarkCards: Standardized Documentation for Large Language Model Benchmarks. arXiv:2410.12974; https://papers.neurips.cc/paper_files/paper/2025/hash/76175f4355e2f67cf91be468c8860070-Abstract-Datasets_and_Benchmarks_Track.html (peer reviewed).
- `bordes2025evalfacts` — Eval Factsheets: A Structured Framework for Documenting AI Evaluations. arXiv:2512.04062; https://arxiv.org/abs/2512.04062 (preprint).
- `oh2026sota` — State-of-the-Art Claims Require State-of-the-Art Evidence. arXiv:2605.17273; https://arxiv.org/abs/2605.17273 (preprint).
- `saito2015pr` — The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. 10.1371/journal.pone.0118432; https://doi.org/10.1371/journal.pone.0118432 (peer reviewed).
- `guo2017calibration` — On Calibration of Modern Neural Networks. PMLR 70; https://proceedings.mlr.press/v70/guo17a.html (peer reviewed).
- `saerens2002prior` — Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure. 10.1162/089976602753284446; https://doi.org/10.1162/089976602753284446 (peer reviewed).
- `geifman2017selective` — Selective Classification for Deep Neural Networks. official proceedings; https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html (peer reviewed).
- `bahnsen2015cost` — Example-Dependent Cost-Sensitive Decision Trees. 10.1016/j.eswa.2015.04.042; https://doi.org/10.1016/j.eswa.2015.04.042 (peer reviewed).
- `siddiqui2018feedback` — Feedback-Guided Anomaly Discovery via Online Optimization. 10.1145/3219819.3220083; https://doi.org/10.1145/3219819.3220083 (peer reviewed).
- `kipf2017gcn` — Semi-Supervised Classification with Graph Convolutional Networks. arXiv:1609.02907; https://arxiv.org/abs/1609.02907 (peer reviewed).
- `hamilton2017graphsage` — Inductive Representation Learning on Large Graphs. arXiv:1706.02216; https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs (peer reviewed).
- `velickovic2018gat` — Graph Attention Networks. arXiv:1710.10903; https://arxiv.org/abs/1710.10903 (peer reviewed).
- `xu2019gin` — How Powerful are Graph Neural Networks?. arXiv:1810.00826; https://arxiv.org/abs/1810.00826 (peer reviewed).
- `hu2020pretraining` — Strategies for Pre-training Graph Neural Networks. arXiv:1905.12265; https://openreview.net/forum?id=HJlWWJSFDH (peer reviewed).
- `friedman2001gbm` — Greedy Function Approximation: A Gradient Boosting Machine. 10.1214/aos/1013203451; https://doi.org/10.1214/aos/1013203451 (peer reviewed).
- `holm1979` — A Simple Sequentially Rejective Multiple Test Procedure. JSTOR 4615733; https://www.jstor.org/stable/4615733 (peer reviewed).
- `benjamini1995` — Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. 10.1111/j.2517-6161.1995.tb02031.x; https://doi.org/10.1111/j.2517-6161.1995.tb02031.x (peer reviewed).
- `efron1979bootstrap` — Bootstrap Methods: Another Look at the Jackknife. 10.1214/aos/1176344552; https://doi.org/10.1214/aos/1176344552 (peer reviewed).
