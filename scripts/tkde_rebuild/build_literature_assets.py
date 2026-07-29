#!/usr/bin/env python3
"""Generate verified literature, novelty, and paper-identity assets."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tkde_rebuild"

FIELDS = [
    "category",
    "cite_key",
    "title",
    "authors",
    "venue",
    "year",
    "doi_or_arxiv",
    "official_url",
    "publication_status",
    "relevance",
    "differentiation",
    "verification_status",
]


def r(category: str, key: str, title: str, authors: str, venue: str, year: int, identifier: str, url: str, status: str, relevance: str, differentiation: str) -> dict[str, str | int]:
    return {
        "category": category,
        "cite_key": key,
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": year,
        "doi_or_arxiv": identifier,
        "official_url": url,
        "publication_status": status,
        "relevance": relevance,
        "differentiation": differentiation,
        "verification_status": "VERIFIED_PRIMARY_OR_OFFICIAL",
    }


REFERENCES = [
    r("graph fraud and AML", "weber2019aml", "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", "Mark Weber; Giacomo Domeniconi; Jie Chen; Daniel Karl I. Weidele; Claudio Bellei; Tom Robinson; Charles E. Leiserson", "KDD 2019 Workshop on Anomaly Detection in Finance; arXiv", 2019, "arXiv:1908.02591", "https://arxiv.org/abs/1908.02591", "workshop/preprint", "Elliptic source and original graph-fraud baseline study.", "No multi-axis deployment contract, paired visibility study, review budget, or claim support relation."),
    r("graph fraud and AML", "liu2020graphconsis", "Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection", "Zhiwei Liu; Yingtong Dou; Philip S. Yu; Yutong Deng; Hao Peng", "SIGIR", 2020, "10.1145/3397271.3401253", "https://doi.org/10.1145/3397271.3401253", "peer reviewed", "GraphConsis addresses context, feature, and relation inconsistency.", "Detector contribution; does not benchmark temporal visibility or claim scope."),
    r("graph fraud and AML", "dou2020caregnn", "Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters", "Yingtong Dou; Zhiwei Liu; Li Sun; Yutong Deng; Hao Peng; Philip S. Yu", "CIKM", 2020, "10.1145/3340531.3411903", "https://doi.org/10.1145/3340531.3411903", "peer reviewed", "CARE-GNN treats feature and relation camouflage.", "Model-centered evaluation rather than a deployment benchmark."),
    r("graph fraud and AML", "liu2021pcgnn", "Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection", "Yang Liu; Xiang Ao; Zidi Qin; Jianfeng Chi; Jinghua Feng; Hao Yang; Qing He", "The Web Conference", 2021, "10.1145/3442381.3449989", "https://doi.org/10.1145/3442381.3449989", "peer reviewed", "PC-GNN addresses imbalance and neighborhood sampling.", "Does not compare graph-visibility contracts or rank-versus-decision conclusions."),
    r("graph anomaly benchmarks", "tang2022bwgnn", "Rethinking Graph Neural Networks for Anomaly Detection", "Jianheng Tang; Jiajin Li; Ziqi Gao; Jia Li", "ICML", 2022, "PMLR 162", "https://proceedings.mlr.press/v162/tang22b.html", "peer reviewed", "BWGNN is an anomaly-specific spectral graph model.", "Static/model contribution; no fraud deployment contract."),
    r("datasets and benchmarks", "huang2022dgraph", "DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection", "Xuanwen Huang; Yang Yang; Yang Wang; Chunping Wang; Zhisheng Zhang; Jiarong Xu; Lei Chen; Michalis Vazirgiannis", "NeurIPS Datasets and Benchmarks", 2022, "10.52202/068431-1654", "https://proceedings.neurips.cc/paper_files/paper/2022/hash/8f1918f71972789db39ec0d85bb31110-Abstract-Datasets_and_Benchmarks.html", "peer reviewed", "Canonical real financial graph and DGraphFin source.", "Introduces data and baselines; does not vary deployment-contract axes."),
    r("graph anomaly benchmarks", "tang2023gadbench", "GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection", "Jianheng Tang; Fengrui Hua; Ziqi Gao; Peilin Zhao; Jia Li", "NeurIPS Datasets and Benchmarks", 2023, "10.52202/075280-1289", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html", "peer reviewed", "Broad supervised graph anomaly benchmark and evidence for strong tree baselines.", "Predominantly static; no temporal visibility, review capacity, or executable claim support."),
    r("datasets and benchmarks", "altman2023ibmaml", "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", "Erik Altman; Jovan Blanuša; Luc von Niederhäusern; Béni Egressy; Andreea Anghel; Kubilay Atasu", "NeurIPS Datasets and Benchmarks", 2023, "10.52202/075280-1300", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html", "peer reviewed", "Canonical IBM AML-Data generator and variant source.", "Dataset generation work; does not establish graph-construction or claim-evidence governance."),
    r("graph fraud and AML", "egressy2024directed", "Provably Powerful Graph Neural Networks for Directed Multigraphs", "Béni Egressy; Luc von Niederhäusern; Jovan Blanuša; Erik Altman; Roger Wattenhofer; Kubilay Atasu", "AAAI", 2024, "10.1609/aaai.v38i10.29069", "https://ojs.aaai.org/index.php/AAAI/article/view/29069", "peer reviewed", "Directed-multigraph expressiveness for financial crime graphs.", "Model expressiveness rather than evaluation-contract completeness."),
    r("graph fraud and AML", "wu2023grande", "GRANDE: a neural model over directed multigraphs with application to anti-money laundering", "Ruofan Wu; Boqun Ma; Hong Jin; Wenlong Zhao; Weiqiang Wang; Tianyi Zhang", "arXiv", 2023, "arXiv:2302.02101", "https://arxiv.org/abs/2302.02101", "preprint", "Directed multigraph edge-classification model for AML.", "No verified peer-reviewed version; not a deployment benchmark."),
    r("graph fraud and AML", "lin2024fraudgt", "FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection", "Junhong Lin; Xiaojie Guo; Yada Zhu; Samuel Mitchell; Erik Altman; Julian Shun", "ICAIF", 2024, "10.1145/3677052.3698648", "https://research.ibm.com/publications/fraudgt-a-simple-effective-and-efficient-graph-transformer-for-financial-fraud-detection", "peer reviewed", "Financial graph-transformer model emphasizing effectiveness and throughput.", "Model comparison; does not define temporal visibility or evidence support."),
    r("graph fraud and AML", "blanusa2024gfp", "Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection", "Jovan Blanuša; Maximo Cravero Baraja; Andreea Anghel; Luc von Niederhäusern; Erik Altman; Haris Pozidis; Kubilay Atasu", "ICAIF", 2024, "10.1145/3677052.3698674", "https://authors.library.caltech.edu/records/vav4g-hxz75", "peer reviewed", "Graph-derived features with boosted trees and throughput analysis.", "Supports strong tabular baselines and resource analysis; lacks claim-support formalism."),
    r("dynamic anomaly benchmarks", "hua2026bag", "BAG: Benchmarking Anomaly Detection on Dynamic Graphs", "Fengrui Hua; Yiyan Qi; Zikai Wei; Yuxing Tian; Chengjin Xu; Xiaojun Wu; Jia Li; Jian Guo", "AAAI", 2026, "10.1609/aaai.v40i17.38510", "https://ojs.aaai.org/index.php/AAAI/article/view/38510", "peer reviewed", "Closest peer-reviewed dynamic graph anomaly benchmark.", "Precludes a first-dynamic-benchmark claim; does not combine fraud-specific visibility, review budgets, and claim support."),
    r("dynamic anomaly benchmarks", "zhou2026wild", "GAD in the Wild: Benchmarking Graph Anomaly Detection under Realistic Deployment Challenges", "Jingjing Zhou; Shiyu Huang; Qing Qing; Zuquan Yuan; Huafei Huang; Ziqi Xu; Mingliang Hou; Xikun Zhang; Renqiang Luo; Ivan Lee", "arXiv", 2026, "arXiv:2605.07133", "https://arxiv.org/abs/2605.07133", "preprint", "Deployment-oriented graph anomaly benchmark covering scale, rarity, and missing attributes.", "Closest positioning threat; named challenges differ from temporal visibility, decision capacity, and support relations."),
    r("fraud surveys", "pourhabibi2020survey", "Fraud Detection: A Systematic Literature Review of Graph-Based Anomaly Detection Approaches", "Tahereh Pourhabibi; Kok-Leong Ong; Booi H. Kam; Yee Ling Boo", "Decision Support Systems", 2020, "10.1016/j.dss.2020.113303", "https://doi.org/10.1016/j.dss.2020.113303", "peer reviewed", "Domain taxonomy and credibility challenges for graph-based fraud detection.", "Survey rather than executable benchmark."),
    r("temporal graph benchmarks", "hu2020ogb", "Open Graph Benchmark: Datasets for Machine Learning on Graphs", "Weihua Hu; Matthias Fey; Marinka Zitnik; Yuxiao Dong; Hongyu Ren; Bowen Liu; Michele Catasta; Jure Leskovec", "NeurIPS", 2020, "official proceedings", "https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html", "peer reviewed", "Standardized datasets, splits, metrics, and scalable evaluation.", "Broad and mostly static; no fraud-specific contract or support relation."),
    r("temporal graph benchmarks", "poursafaei2022better", "Towards Better Evaluation for Dynamic Link Prediction", "Farimah Poursafaei; Shenyang Huang; Kellin Pelrine; Reihaneh Rabbany", "NeurIPS Datasets and Benchmarks", 2022, "10.52202/068431-2386", "https://papers.neurips.cc/paper_files/paper/2022/hash/d49042a5d49818711c401d34172f9900-Abstract-Datasets_and_Benchmarks.html", "peer reviewed", "Shows temporal evaluation choices can inflate results and alter rankings.", "Link prediction rather than extreme-imbalance fraud screening."),
    r("temporal graph benchmarks", "huang2023tgb", "Temporal Graph Benchmark for Machine Learning on Temporal Graphs", "Shenyang Huang; Farimah Poursafaei; Jacob Danovitch; Matthias Fey; Weihua Hu; Emanuele Rossi; Jure Leskovec; Michael Bronstein; Guillaume Rabusseau; Reihaneh Rabbany", "NeurIPS Datasets and Benchmarks", 2023, "10.52202/075280-0099", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/066b98e63313162f6562b35962671288-Abstract-Datasets_and_Benchmarks.html", "peer reviewed", "Standardizes large temporal tasks and realistic protocols.", "Does not cover fraud graph construction, review capacity, or scoped evidence completeness."),
    r("temporal graph benchmarks", "gastinger2024tgb2", "TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs", "Julia Gastinger; Shenyang Huang; Mikhail Galkin; Erfan Loghmani; Ali Parviz; Farimah Poursafaei; Jacob Danovitch; Emanuele Rossi; Ioannis Koutis; Heiner Stuckenschmidt; Reihaneh Rabbany; Guillaume Rabusseau", "NeurIPS Datasets and Benchmarks", 2024, "10.52202/079017-4450", "https://proceedings.neurips.cc/paper_files/paper/2024/hash/fda026cf2423a01fcbcf1e1e43ee9a50-Abstract-Datasets_and_Benchmarks_Track.html", "peer reviewed", "Extends temporal evaluation to heterogeneous and knowledge graphs and reports scale failures.", "Resource visibility itself is not novel; status semantics and claim handling remain differentiators."),
    r("temporal graph benchmarks", "huang2024benchtemp", "BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks", "Qiang Huang; Xin Wang; Susie Xi Rao; Zhichao Han; Zitao Zhang; Yongjun He; Quanqing Xu; Yang Zhao; Zhigao Zheng; Jiawei Jiang", "ICDE", 2024, "10.1109/ICDE60146.2024.00310", "https://arxiv.org/abs/2308.16385", "peer reviewed", "Unifies temporal datasets, training, inductive/transductive evaluation, and efficiency.", "Precludes a broad first-protocol-aware benchmark claim."),
    r("distribution shift", "gui2022good", "GOOD: A Graph Out-of-Distribution Benchmark", "Shurui Gui; Xiner Li; Limei Wang; Shuiwang Ji", "NeurIPS Datasets and Benchmarks", 2022, "10.52202/068431-0150", "https://proceedings.neurips.cc/paper_files/paper/2022/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html", "peer reviewed", "Separates covariate and concept shift on graph and node tasks.", "Controlled OOD domains differ from temporal graph availability and operational budgets."),
    r("evaluation validity", "cawley2010selection", "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation", "Gavin C. Cawley; Nicola L. C. Talbot", "Journal of Machine Learning Research", 2010, "JMLR 11", "https://jmlr.org/beta/papers/v11/cawley10a.html", "peer reviewed", "Foundational separation of model selection and final evaluation.", "Motivates selection cleanliness but does not instantiate a fraud benchmark."),
    r("evaluation validity", "kaufman2012leakage", "Leakage in Data Mining: Formulation, Detection, and Avoidance", "Shachar Kaufman; Saharon Rosset; Claudia Perlich; Ori Stitelman", "ACM Transactions on Knowledge Discovery from Data", 2012, "10.1145/2382577.2382579", "https://doi.org/10.1145/2382577.2382579", "peer reviewed", "Formalizes leakage as use of information unavailable at legitimate prediction time.", "Does not operationalize a multi-axis temporal fraud contract."),
    r("evaluation validity", "kapoor2023leakage", "Leakage and the Reproducibility Crisis in Machine-Learning-Based Science", "Sayash Kapoor; Arvind Narayanan", "Patterns", 2023, "10.1016/j.patter.2023.100804", "https://doi.org/10.1016/j.patter.2023.100804", "peer reviewed", "Leakage taxonomy and reproducibility guidance.", "Governance guidance, not a run-level claim-evidence validator."),
    r("evaluation validity", "gorman2019splits", "We Need to Talk about Standard Splits", "Kyle Gorman; Steven Bedrick", "ACL", 2019, "10.18653/v1/P19-1267", "https://aclanthology.org/P19-1267/", "peer reviewed", "Demonstrates ranking sensitivity to data splits.", "Not graph- or fraud-specific and does not define deployment estimands."),
    r("documentation and governance", "bender2018data", "Data Statements for Natural Language Processing", "Emily M. Bender; Batya Friedman", "Transactions of the Association for Computational Linguistics", 2018, "10.1162/tacl_a_00041", "https://doi.org/10.1162/tacl_a_00041", "peer reviewed", "Links documented data scope to defensible generalization.", "Descriptive documentation rather than executable evidence completeness."),
    r("documentation and governance", "mitchell2019modelcards", "Model Cards for Model Reporting", "Margaret Mitchell; Simone Wu; Andrew Zaldivar; Parker Barnes; Lucy Vasserman; Ben Hutchinson; Elena Spitzer; Inioluwa Deborah Raji; Timnit Gebru", "FAT*", 2019, "10.1145/3287560.3287596", "https://doi.org/10.1145/3287560.3287596", "peer reviewed", "Documents intended use and disaggregated performance.", "Does not define completeness conditions for a scoped empirical claim."),
    r("documentation and governance", "gebru2021datasheets", "Datasheets for Datasets", "Timnit Gebru; Jamie Morgenstern; Briana Vecchione; Jennifer Wortman Vaughan; Hanna Wallach; Hal Daumé III; Kate Crawford", "Communications of the ACM", 2021, "10.1145/3458723", "https://doi.org/10.1145/3458723", "peer reviewed", "Dataset motivation, composition, collection, and use documentation.", "Dataset cards are prior art, not the benchmark's novelty."),
    r("documentation and governance", "pushkarna2022datacards", "Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI", "Mahima Pushkarna; Andrew Zaldivar; Oddur Kjartansson", "FAccT", 2022, "10.1145/3531146.3533231", "https://doi.org/10.1145/3531146.3533231", "peer reviewed", "Stakeholder- and lifecycle-aware dataset documentation.", "Descriptive, not a claim-to-evidence support relation."),
    r("construct validity", "jacobs2021measurement", "Measurement and Fairness", "Abigail Z. Jacobs; Hanna Wallach", "FAccT", 2021, "10.1145/3442188.3445901", "https://doi.org/10.1145/3442188.3445901", "peer reviewed", "Construct reliability and validity for sociotechnical measurement.", "Provides language for metric validity; no fraud benchmark implementation."),
    r("benchmark governance", "dehghani2021lottery", "The Benchmark Lottery", "Mostafa Dehghani; Yi Tay; Alexey A. Gritsenko; Zhe Zhao; Neil Houlsby; Fernando Diaz; Donald Metzler; Oriol Vinyals", "arXiv/OpenReview", 2021, "arXiv:2107.07002", "https://arxiv.org/abs/2107.07002", "preprint", "Argues that benchmark and task choices alter perceived model superiority.", "Conceptual precedent; no executable evidence support."),
    r("benchmark governance", "reuel2024betterbench", "BetterBench: Assessing AI Benchmarks, Uncovering Issues, and Establishing Best Practices", "Anka Reuel; Amelia Hardy; Chandler Smith; Max Lamparth; Malcolm Hardy; Mykel J. Kochenderfer", "NeurIPS Datasets and Benchmarks", 2024, "10.52202/079017-0685", "https://proceedings.neurips.cc/paper_files/paper/2024/hash/26889e8359e7ef8a7f5d77457364ca55-Abstract-Datasets_and_Benchmarks_Track.html", "peer reviewed", "Benchmark lifecycle quality assessment and best practices.", "Evaluates benchmark quality, not support for a particular scoped result claim."),
    r("benchmark governance", "sokol2025benchmarkcards", "BenchmarkCards: Standardized Documentation for Large Language Model Benchmarks", "Anna Sokol; Elizabeth Daly; Michael Hind; David Piorkowski; Xiangliang Zhang; Nuno Moniz; Nitesh V. Chawla", "NeurIPS Datasets and Benchmarks", 2025, "arXiv:2410.12974", "https://papers.neurips.cc/paper_files/paper/2025/hash/76175f4355e2f67cf91be468c8860070-Abstract-Datasets_and_Benchmarks_Track.html", "peer reviewed", "Standardizes benchmark metadata and selection.", "Does not prescribe run-level claim support or blocked-status semantics."),
    r("benchmark governance", "bordes2025evalfacts", "Eval Factsheets: A Structured Framework for Documenting AI Evaluations", "Florian Bordes; Candace Ross; Justine T. Kao; Evangelia Spiliopoulou; Adina Williams", "arXiv", 2025, "arXiv:2512.04062", "https://arxiv.org/abs/2512.04062", "preprint", "Structures evaluation context, scope, construction, method, and alignment.", "Descriptive; FraudShiftBench must distinguish executable completeness and status semantics."),
    r("benchmark governance", "oh2026sota", "State-of-the-Art Claims Require State-of-the-Art Evidence", "YongKyung Oh", "arXiv", 2026, "arXiv:2605.17273", "https://arxiv.org/abs/2605.17273", "preprint", "Directly questions whether aggregate evidence supports superiority claims.", "Claim gating alone is insufficient novelty; no fraud deployment contract."),
    r("imbalance and metrics", "saito2015pr", "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets", "Takaya Saito; Marc Rehmsmeier", "PLOS ONE", 2015, "10.1371/journal.pone.0118432", "https://doi.org/10.1371/journal.pone.0118432", "peer reviewed", "Canonical motivation for precision-recall analysis under imbalance.", "Does not replace operational Precision@K/Recall@K."),
    r("calibration", "guo2017calibration", "On Calibration of Modern Neural Networks", "Chuan Guo; Geoff Pleiss; Yu Sun; Kilian Q. Weinberger", "ICML", 2017, "PMLR 70", "https://proceedings.mlr.press/v70/guo17a.html", "peer reviewed", "Temperature scaling baseline and neural calibration evidence.", "Monotone calibration changes threshold decisions but not rank order."),
    r("calibration", "saerens2002prior", "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure", "Marco Saerens; Patrice Latinne; Christine Decaestecker", "Neural Computation", 2002, "10.1162/089976602753284446", "https://doi.org/10.1162/089976602753284446", "peer reviewed", "Prior-probability-shift correction.", "Does not justify improvement under arbitrary temporal shift."),
    r("selective prediction", "geifman2017selective", "Selective Classification for Deep Neural Networks", "Yonatan Geifman; Ran El-Yaniv", "NeurIPS", 2017, "official proceedings", "https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html", "peer reviewed", "Risk-coverage formalism for abstaining classifiers.", "Different from a fixed analyst budget over ranked fraud alerts."),
    r("cost-sensitive learning", "bahnsen2015cost", "Example-Dependent Cost-Sensitive Decision Trees", "Alejandro Correa Bahnsen; Djamila Aouada; Björn Ottersten", "Expert Systems with Applications", 2015, "10.1016/j.eswa.2015.04.042", "https://doi.org/10.1016/j.eswa.2015.04.042", "peer reviewed", "Explicit decision costs in fraud-related classification.", "Does not address temporal graph evaluation contracts."),
    r("human review", "siddiqui2018feedback", "Feedback-Guided Anomaly Discovery via Online Optimization", "Md Amran Siddiqui; Alan Fern; Thomas G. Dietterich; Ryan Wright; Alec Theriault; David W. Archer", "KDD", 2018, "10.1145/3219819.3220083", "https://doi.org/10.1145/3219819.3220083", "peer reviewed", "Human analyst inspects a ranked anomaly list under limited effort.", "Operational precedent for review budgets, but not temporal graph fraud."),
    r("graph models", "kipf2017gcn", "Semi-Supervised Classification with Graph Convolutional Networks", "Thomas N. Kipf; Max Welling", "ICLR", 2017, "arXiv:1609.02907", "https://arxiv.org/abs/1609.02907", "peer reviewed", "Canonical GCN architecture.", "Model citation only."),
    r("graph models", "hamilton2017graphsage", "Inductive Representation Learning on Large Graphs", "William L. Hamilton; Rex Ying; Jure Leskovec", "NeurIPS", 2017, "arXiv:1706.02216", "https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs", "peer reviewed", "Canonical GraphSAGE architecture.", "Model citation only."),
    r("graph models", "velickovic2018gat", "Graph Attention Networks", "Petar Veličković; Guillem Cucurull; Arantxa Casanova; Adriana Romero; Pietro Liò; Yoshua Bengio", "ICLR", 2018, "arXiv:1710.10903", "https://arxiv.org/abs/1710.10903", "peer reviewed", "Canonical graph attention architecture.", "Model citation only."),
    r("graph models", "xu2019gin", "How Powerful are Graph Neural Networks?", "Keyulu Xu; Weihua Hu; Jure Leskovec; Stefanie Jegelka", "ICLR", 2019, "arXiv:1810.00826", "https://arxiv.org/abs/1810.00826", "peer reviewed", "Canonical GIN architecture underlying GINE.", "Model citation only."),
    r("graph models", "hu2020pretraining", "Strategies for Pre-training Graph Neural Networks", "Weihua Hu; Bowen Liu; Joseph Gomes; Marinka Zitnik; Percy Liang; Vijay Pande; Jure Leskovec", "ICLR", 2020, "arXiv:1905.12265", "https://openreview.net/forum?id=HJlWWJSFDH", "peer reviewed", "Primary source for the edge-conditioned GINE update used by the repository implementation.", "Model citation only; the benchmark does not study pre-training."),
    r("tabular models", "friedman2001gbm", "Greedy Function Approximation: A Gradient Boosting Machine", "Jerome H. Friedman", "The Annals of Statistics", 2001, "10.1214/aos/1013203451", "https://doi.org/10.1214/aos/1013203451", "peer reviewed", "Foundational gradient boosting method.", "Model citation only."),
    r("statistics", "holm1979", "A Simple Sequentially Rejective Multiple Test Procedure", "Sture Holm", "Scandinavian Journal of Statistics", 1979, "JSTOR 4615733", "https://www.jstor.org/stable/4615733", "peer reviewed", "Family-wise error correction used in paired comparisons.", "Statistical method citation."),
    r("statistics", "benjamini1995", "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing", "Yoav Benjamini; Yosef Hochberg", "Journal of the Royal Statistical Society, Series B", 1995, "10.1111/j.2517-6161.1995.tb02031.x", "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x", "peer reviewed", "False-discovery-rate correction for supplement diagnostics.", "Statistical method citation."),
    r("statistics", "efron1979bootstrap", "Bootstrap Methods: Another Look at the Jackknife", "Bradley Efron", "The Annals of Statistics", 1979, "10.1214/aos/1176344552", "https://doi.org/10.1214/aos/1176344552", "peer reviewed", "Bootstrap interval foundation.", "Statistical method citation."),
]


NOVELTY_FIELDS = ["work", "domain", "temporal_protocols", "graph_visibility_controls", "graph_construction_ablations", "class_prior_regimes", "review_budget_metrics", "resource_status_semantics", "prediction_provenance", "machine_checkable_claim_support", "fraudshiftbench_difference"]
NOVELTY_ROWS = [
    ["GADBench (2023)", "static supervised graph anomaly", "limited", "no", "limited", "imbalance-aware evaluation", "no", "no", "benchmark code/data", "no", "FraudShiftBench focuses temporal fraud contracts, decisions, resources, and scoped support."],
    ["BAG (2026)", "dynamic graph anomaly", "yes", "not fraud-specific", "model/task breadth", "not a central axis", "no", "reports scale but no blocked ordering semantics", "benchmark outputs", "no", "Closest peer-reviewed dynamic benchmark; proposed combination remains different."],
    ["GAD in the Wild (2026)", "deployment graph anomaly", "some deployment axes", "not the proposed contract", "missing-attribute/scale stress", "extreme rarity", "no fixed analyst contract", "scale challenge", "not a central contribution", "no", "Preprint and closest positioning threat; do not claim first realistic deployment benchmark."],
    ["TGB (2023) / TGB 2.0 (2024)", "temporal graph tasks", "yes", "task-specific inductive protocols", "not fraud construction grid", "no fraud-prior regimes", "no", "scale failures visible", "standardized data/results", "no", "FraudShiftBench specializes in fraud units, imbalance, graph construction, budgets, and evidence scope."],
    ["BenchTemp (2024)", "temporal GNN evaluation", "transductive/inductive", "yes", "not fraud-specific", "no", "no", "efficiency evaluation", "benchmark pipeline", "no", "Precludes broad protocol/efficiency-first claims."],
    ["GOOD (2022)", "graph OOD", "controlled domains", "no", "no", "not fraud-specific", "no", "no", "benchmark splits", "no", "Controlled covariate/concept shift differs from deployment visibility and analyst constraints."],
    ["DGraph (2022)", "real financial node anomaly", "dynamic data", "no systematic grid", "no", "rare labels/background", "no", "scale discussed", "dataset access", "no", "FraudShiftBench uses DGraphFin to test contract sensitivity rather than proposing a new detector."],
    ["IBM AML-Data (2023)", "synthetic AML transactions", "timestamps/variants", "no", "generator/baselines", "HI/LI variants", "no", "dataset scale variants", "data provenance", "no", "FraudShiftBench treats graph construction, decisions, and blocked scale explicitly while retaining synthetic limitation."],
    ["FraudGT / directed-multigraph GNNs", "financial graph models", "model-specific", "fixed evaluation", "model-defined", "imbalance present", "throughput", "efficiency", "implementation outputs", "no", "FraudShiftBench evaluates conclusion sensitivity rather than claiming model novelty."],
    ["Graph Feature Preprocessor (2024)", "financial graph-derived features", "not central", "fixed", "subgraph features", "fraud imbalance", "operational throughput", "yes", "experiment artifacts", "no", "Strong adjacent resource/baseline work; support-relation and contract combination remains distinct."],
    ["BetterBench (2024)", "benchmark governance", "generic", "generic checklist", "generic", "generic", "generic", "lifecycle practices", "documentation", "no per-claim validator", "FraudShiftBench instantiates an executable, typed support relation for one high-risk domain."],
    ["BenchmarkCards / Eval Factsheets", "evaluation documentation", "documented", "documented", "documented", "documented", "documented", "documented", "metadata-centric", "descriptive, not run-level completeness", "FraudShiftBench builds on documentation and tests evidence completeness/status transitions."],
]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(clean(row[field]) for field in fields) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "LITERATURE_MATRIX.csv", REFERENCES, FIELDS)
    novelty = [dict(zip(NOVELTY_FIELDS, row)) for row in NOVELTY_ROWS]
    write_csv(OUT / "NOVELTY_DIFFERENTIATION_TABLE.csv", novelty, NOVELTY_FIELDS)

    preprints = [row for row in REFERENCES if row["publication_status"] in {"preprint", "workshop/preprint"}]
    verification_lines = "\n".join(
        f"- `{row['cite_key']}` — {row['title']}. {row['doi_or_arxiv']}; {row['official_url']} ({row['publication_status']})."
        for row in REFERENCES
    )
    preprint_lines = "\n".join(f"- `{row['cite_key']}` must be described as {row['publication_status']}." for row in preprints)
    (OUT / "CITATION_VERIFICATION.md").write_text(
        f"""# Citation Verification

All **{len(REFERENCES)}** entries in `LITERATURE_MATRIX.csv` were checked against a DOI resolver, official proceedings/journal page, PMLR/JMLR/ACL anthology record, or the cited arXiv record. The matrix records publication status so preprints are not presented as peer-reviewed work. No bibliography entry may be inserted unless it maps to one of these verified records or is added with the same verification fields.

## Status cautions

{preprint_lines}

- Elliptic has no verified archival ACM DOI; cite the 2019 arXiv/workshop paper.
- GRANDE has no verified peer-reviewed version in this audit.
- OGB and BenchmarkCards have official proceedings pages but no verified paper DOI in the inspected record.
- Do not reproduce unstable BetterBench checklist counts; title, authors, venue, year, DOI, and conceptual relevance are stable.
- Do not claim that FraudShiftBench is the first dynamic, protocol-aware, resource-aware, or deployment-oriented graph-anomaly benchmark.

## Verified entries

{verification_lines}
""",
        encoding="utf-8",
    )

    synthesis = """# Literature Synthesis and Defensible Gap

## Fraud and graph anomaly detection

The fraud literature has largely treated evaluation as the backdrop to a detector contribution. Elliptic established a temporal Bitcoin benchmark, while GraphConsis, CARE-GNN, PC-GNN, BWGNN, directed-multigraph GNNs, GRANDE, and FraudGT each target a particular modeling failure or representation. This work is indispensable for model selection, but a strong result within one graph construction and one split does not show that the conclusion survives a different deployment contract. Two adjacent findings are especially important for the present study: GADBench shows that broad graph-anomaly comparisons can favor tree ensembles, and the Graph Feature Preprocessor shows that graph-derived features plus boosted trees can be both accurate and efficient. FraudShiftBench therefore treats strong non-graph and graph-derived tabular methods as substantive baselines rather than foils.

The benchmark landscape has also moved beyond static anomaly detection. DGraph supplies a large real financial graph; IBM AML-Data supplies synthetic, controlled AML regimes; BAG evaluates dynamic graph anomaly detection at breadth; and GAD in the Wild studies deployment challenges such as scale, rarity, and missing attributes. These works rule out a credible claim that FraudShiftBench is the first dynamic or deployment-oriented anomaly benchmark. The remaining gap is narrower: none of the inspected benchmarks combines fraud-specific temporal units, graph visibility and construction, model-selection cleanliness, investigation capacity, resource-status semantics, prediction provenance, and a typed support test for every scoped conclusion.

## Temporal evaluation and shift

OGB, GOOD, the dynamic-link evaluation study, TGB, TGB 2.0, and BenchTemp demonstrate that splits, negative sampling, inductive assumptions, temporal order, scale, and efficiency can change benchmark conclusions. They establish the premise that an evaluation protocol is part of the estimand. Fraud screening adds constraints these benchmarks do not jointly encode: graph edges may be available on a different schedule than labels, the positive prior is extremely small and time-varying, analysts inspect only a small prefix of a ranked list, and some graph constructions fail within the declared resource envelope. FraudShiftBench specializes the general protocol lesson into a deployment contract for this domain.

## Validity, documentation, and claim scope

Selection bias and leakage research explains why validation cleanliness and prediction-time information must be explicit. Data statements, model cards, datasheets, data cards, BenchmarkCards, Eval Factsheets, and BetterBench provide mature documentation and governance precedents. The contribution here is not another checklist or dataset card. The proposed evidence unit links the deployment contract, model/configuration, seeds, metrics, prediction manifest, resource record, and integrity state. A typed claim is supported only when its required evidence units are complete, paired when needed, and valid for the requested scope. The claim gate is an implementation of that relation, not the central scientific novelty.

## Decision metrics under extreme imbalance

Precision-recall analysis is better aligned with rare positives than ROC analysis alone, yet AUPRC still summarizes ranking rather than an operational decision. Calibration and prior correction can change probabilities and thresholded decisions, while strictly monotone calibration cannot change ranking metrics. Selective classification studies risk at varying coverage; cost-sensitive learning attaches explicit costs; feedback-guided anomaly discovery models a human inspecting a ranked list. FraudShiftBench brings these ideas into temporal graph-fraud evaluation through fixed review budgets, cost-sensitive risk, and a strict separation between rank and decision claims.

## Positioning conclusion

The defensible novelty is combinatorial but technically precise: a fraud-specific deployment contract over independent evaluation axes, coupled to a machine-checkable claim-evidence support relation and instantiated on prediction-backed, ten-seed evidence with visible resource boundaries. The empirical contribution is a conditional map, not a universal verdict: graph access and construction can help, be neutral, or hurt depending on dataset, model, temporal regime, metric, scale, and feasible resource set. GraphSafe-TTA is retained only as a bounded example of how rank and decision conclusions diverge under the framework.
"""
    (OUT / "LITERATURE_SYNTHESIS.md").write_text(synthesis, encoding="utf-8")

    identity = """# Paper Identity Decision

## Selected identity

**Title:** *FraudShiftBench: Deployment-Contract Evaluation for Temporal Graph Fraud Detection*

**Thesis:** A model result in temporal graph fraud detection is meaningful only relative to a deployment contract that fixes time, graph visibility, construction, selection, decision capacity, and resource envelope. Across locked ten-seed evidence on Elliptic, DGraphFin, and synthetic IBM AML-Data, model rankings and operational conclusions are conditional on those choices. A typed claim-evidence relation prevents conclusions from being widened beyond complete, prediction-backed evidence.

**Contribution hierarchy:**

1. The primary methodological contribution is the deployment contract and its fraud-specific instantiation.
2. The support relation and validator are enabling technical infrastructure that makes claim scope executable.
3. The primary empirical contribution is a conditional map of protocol, architecture, construction, metric, scale, and resource interactions across two real datasets and controlled synthetic AML regimes.
4. The auditable artifact includes prediction manifests, paired seeds, provenance, deterministic paper regeneration, explicit exclusions, and resource-blocked cells.
5. GraphSafe-TTA is a bounded decision-analysis case study, not a universal method contribution.

## Candidate title scoring

Scores are 1–5. Accuracy and non-overclaim are weighted twice; the selected title has the highest weighted total.

| candidate | accuracy | differentiation | brevity | TKDE fit | non-overclaim | weighted total | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FraudShiftBench: Deployment-Contract Evaluation for Temporal Graph Fraud Detection | 5 | 5 | 5 | 5 | 5 | 35 | selected |
| Beyond Single-Split Leaderboards: Deployment Contracts for Temporal Graph Fraud Detection | 5 | 4 | 4 | 5 | 5 | 33 | strong alternative; less artifact identity |
| Auditing Temporal Graph Fraud Benchmarks with Deployment and Evidence Contracts | 5 | 4 | 4 | 5 | 5 | 33 | sounds narrower and more administrative |
| FraudShiftBench: Protocol-, Construction-, and Resource-Aware Evaluation of Temporal Graph Fraud Detection | 5 | 4 | 2 | 5 | 5 | 31 | accurate but unwieldy |
| From Graph Access to Review Budgets: Evaluating Temporal Fraud Detection under Deployment Contracts | 5 | 4 | 3 | 4 | 5 | 31 | evocative but omits evidence support |
| Scoped Evidence for Temporal Graph Fraud Detection | 4 | 4 | 5 | 4 | 5 | 31 | concise but hides deployment contract |
| When Evaluation Contracts Change the Winner in Temporal Graph Fraud Detection | 4 | 4 | 4 | 5 | 4 | 29 | headline-like and overweights ranking reversal |
| Evaluation Protocols Reverse Model Rankings under Temporal Shift | 3 | 4 | 5 | 4 | 3 | 25 | too universal for mixed DGraphFin/IBM evidence |

## Claims deliberately not made

- first temporal, dynamic, realistic, deployment-oriented, protocol-aware, or resource-aware graph anomaly benchmark;
- universal graph harm, universal GNN failure, or universal GraphSafe-TTA improvement;
- IBM AML Large performance or Medium GINE performance;
- P100, A100, or H100 empirical evidence;
- calibration improvement to AUPRC/AUROC under arbitrary shift;
- statistical significance without an identified test, family, correction, and effect.
"""
    (OUT / "PAPER_IDENTITY_DECISION.md").write_text(identity, encoding="utf-8")
    print(f"wrote {len(REFERENCES)} verified literature records and {len(novelty)} novelty rows")


if __name__ == "__main__":
    main()
