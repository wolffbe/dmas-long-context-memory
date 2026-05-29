# Cost and accuracy of long-term memory in distributed LLM-based multi-agent systems

This repository contains the testbed and the analysis used to produce the empirical results of the paper _Cost and accuracy of long-term memory in distributed multi-agent systems based on large language models_. The paper formulates the research questions, methodology, statistical tests, and discussion; this README only describes what the repository contains and where to find each artefact.

> The paper was submitted to **IEEE COMPSAC 2026**. A link to the publication will be added here once available.

## Overview

When LLM agents need to remember things across long conversations, they rely on a _long-term memory_ system. Several such systems exist, each with a different idea of how memories should be stored and searched. This repository measures how three of them, plus two simple baselines, compare on answer accuracy and operating cost. The benchmark is [LoCoMo](https://github.com/snap-research/locomo), the setting is a realistic cloud-edge deployment.

| Name                                            | How it remembers                                      |
| ----------------------------------------------- | ----------------------------------------------------- |
| [cognee](https://github.com/topoteretes/cognee) | Graph plus vector embeddings, populated by an LLM     |
| [Graphiti](https://github.com/getzep/graphiti)  | A temporal knowledge graph                            |
| [Mem0](https://github.com/mem0ai/mem0)          | LLM-extracted facts in a vector store                 |
| RAG _(baseline)_                                | Raw conversation turns in a vector store, no LLM step |
| full-context _(baseline)_                       | The whole conversation, no compression at all         |

The paper asks whether the extra machinery of a memory framework actually buys better answers, and how that bet shifts when the link between the edge agent and the cloud becomes slow or constrained.

## Repository layout

```
dmas-memory/
├── paper/                  # Manuscript sources
└── testbed/                # Runnable benchmark (see testbed/README.md)
    ├── dmas/                   # Coordinator, responder, memory, benchmark services
    ├── experiments/
    │   ├── results/            # Per-experiment CSV outputs
    │   └── analysis/
    │       ├── results.ipynb       # Reproduces every table and figure of the paper
    │       ├── requirements.txt    # Python dependencies for the notebook
    │       └── figures/            # PDF figures emitted by the notebook
    ├── Makefile
    └── .env.example
```

The benchmark is a Docker Compose stack organised into edge, cloud, and management networks; full operational documentation is in [`testbed/README.md`](testbed/README.md).

## Reproducing the analysis

`testbed/experiments/analysis/results.ipynb` ingests every CSV under `testbed/experiments/results/` and rebuilds, end-to-end, every table and figure reported in the paper: macro retention metrics, retrieval-failure rates, per-category accuracy, the system-level cost decomposition per phase, total cost of ownership, the Pareto frontier, and the per-framework network sensitivity tests.

```bash
cd testbed/experiments/analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook results.ipynb
```

LaTeX strings for the cost and retention tables are printed in place; PDF figures are written to `figures/`.

## Reproducing the experiments

End-to-end execution of the benchmark (building the images, bringing up the stack, running the sweep that produces the CSVs consumed by the notebook above) is documented in [`testbed/README.md`](testbed/README.md), together with the network partitioning, fault-injection and metering details.

## Citation

A BibTeX entry will be provided once the IEEE COMPSAC 2026 publication is available.

LoCoMo benchmark dataset:

```bibtex
@inproceedings{maharana-etal-2024-evaluating,
    title     = "Evaluating Very Long-Term Conversational Memory of {LLM} Agents",
    author    = "Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei",
    editor    = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month     = aug,
    year      = "2024",
    address   = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2024.acl-long.747/",
    doi       = "10.18653/v1/2024.acl-long.747",
    pages     = "13851--13870"
}
```

## License

See [LICENSE.txt](LICENSE.txt).
