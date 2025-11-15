# Readability_review
## Badges 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-ABR-green.svg)](link-to-paper)

# Linguistic Complexity and Financial Distress Prediction
# Evidence from Belgian Corporate Reports

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: CC0](https://img.shields.io/badge/Data-CC0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

Analysis of 116,493 Belgian corporate annual reports using transformer-based natural language processing to predict financial distress.



## Paper

**[Your Name]** and **[Co-author Name]**. (2026). "Linguistic Complexity and Financial Distress: Evidence from Transformer-Based Analysis of Belgian Firms." *Accounting and Business Research*. [DOI: forthcoming]

**Abstract:** This study investigates how corporate disclosure language captures strategic information orthogonal to financial metrics. Analyzing 116,493 Dutch and French annual reports from Belgian firms (2019–2023) using transformer-derived linguistic features and gradient boosting models, we demonstrate that linguistic and financial signals exhibit complementarity rather than redundancy. Linguistic complexity operates through a regime-based structure, with documents clustering into formulaic and narrative modes. Results show transformer-derived features improve bankruptcy prediction, achieving AUC-PR of 0.399 at one-year horizons despite severe class imbalance.

## Repository Structure
```
.
├── README.md                          # This file
├── LICENSE                            # MIT License for code
├── LICENSE_DATA                       # CC0 for processed data
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (optional)
│
├── 01_preprocessing/                  # XBRL parsing & feature extraction
│   ├── README.md                      # Preprocessing documentation
│   ├── parse_xbrl.py                  # XBRL file parsing
│   ├── extract_text.py                # Text extraction from XBRL
│   ├── language_detection.py          # Dutch/French classification
│   ├── feature_extraction_sbert.py    # Sentence-BERT features
│   ├── feature_extraction_llama.py    # LLaMA perplexity/surprisal
│   └── feature_extraction_spacy.py    # Syntactic features
│
├── 02_analysis/                       # Main analysis pipeline
│   ├── README.md                      # Analysis documentation
│   ├── 01_calculate_composites.py     # Composite index calculation
│   ├── 02_regime_clustering.py        # Linguistic regime classification
│   ├── 03_train_models.py             # XGBoost training
│   ├── 04_shap_analysis.py            # Model interpretation
│   ├── 05_temporal_analysis.py        # Structural break detection
│   └── 06_generate_figures.py         # Visualization
│
├── config/                            # Configuration files
│   ├── model_params.yaml              # XGBoost hyperparameters
│   └── paths.yaml                     # Data paths configuration
│
├── docs/                              # Documentation
│   ├── data_dictionary.md             # Variable definitions
│   ├── methodology.md                 # Detailed methodology
│   └── replication_guide.md           # Step-by-step replication
│
└── outputs/                           # Generated outputs (gitignored)
    ├── figures/
    ├── tables/
    └── models/
```

## Data Availability

### Raw Data Sources

**XBRL Annual Reports:**
- **National Bank of Belgium** Central Balance Sheet Office: https://www.nbb.be
- **Bel-First database** (Bureau van Dijk): Institutional subscription required

Our sample comprises 116,493 annual reports (management commentary and auditor reports) from Belgian firms for fiscal years 2019–2023 in Dutch and French.

### Processed Linguistic Features

Due to Bel-First licensing restrictions, raw XBRL files cannot be shared directly. To facilitate replication, we provide processed linguistic features:

**Zenodo Repository:** [DOI: 10.5281/zenodo.XXXXXXX](https://zenodo.org/record/XXXXXXX)

**Dataset includes:**
- Document-level linguistic complexity metrics (150+ features per document)
- Five linguistic dimensions: lexical, syntactic, semantic density, cohesion, cognitive processing
- Nine composite indices: obfuscation, processing difficulty, evasiveness, etc.
- Firm-year panel structure with anonymized firm identifiers
- Complete data dictionary with variable definitions

**Format:** CSV files organized by year  
**Size:** ~25 GB uncompressed (~8 GB compressed)  
**License:** CC0 1.0 Universal (Public Domain Dedication)

**Note:** This dataset contains numerical features derived from text, not verbatim report content.

### Accessing Raw Data

Researchers wishing to replicate from raw XBRL files may:
1. Request access from National Bank of Belgium Central Balance Sheet Office
2. Obtain institutional subscription to Bel-First database
3. Use our documented sampling criteria (see `docs/methodology.md`)

**Processing time:** ~200 hours on high-performance computing infrastructure with GPU acceleration

## Installation

### Prerequisites

- Python 3.9 or higher
- (Optional) NVIDIA GPU with CUDA 11.8+ for preprocessing
- 16 GB RAM minimum (64 GB recommended for preprocessing)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/yourusername/linguistic-distress-prediction.git
cd linguistic-distress-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language models
python -m spacy download nl_core_news_lg
python -m spacy download fr_core_news_lg
```

### Alternative: Conda Environment
```bash
conda env create -f environment.yml
conda activate linguistic-distress
```

## Quick Start

### Using Processed Data (Recommended)

**Estimated time:** 2-4 hours on standard hardware
```bash
# 1. Download processed data from Zenodo
# Place in data/processed/ directory

# 2. Configure paths
cp config/paths.yaml.example config/paths.yaml
# Edit paths.yaml with your local paths

# 3. Run analysis pipeline
python 02_analysis/01_calculate_composites.py
python 02_analysis/02_regime_clustering.py
python 02_analysis/03_train_models.py
python 02_analysis/04_shap_analysis.py
python 02_analysis/05_temporal_analysis.py
python 02_analysis/06_generate_figures.py

# 4. View results
ls outputs/figures/
ls outputs/tables/
```

### Full Reproduction from Raw XBRL (Advanced)

**Estimated time:** 200+ hours

**Requirements:**
- Access to Belgian XBRL filings (see Data Availability above)
- High-performance computing infrastructure
- NVIDIA GPU with 16+ GB VRAM
```bash
# 1. Place raw XBRL files in data/raw/xbrl/

# 2. Run preprocessing pipeline
python 01_preprocessing/parse_xbrl.py --input data/raw/xbrl/ --output data/interim/text/
python 01_preprocessing/language_detection.py
python 01_preprocessing/feature_extraction_sbert.py --gpu  # Requires GPU
python 01_preprocessing/feature_extraction_llama.py --gpu  # Requires GPU
python 01_preprocessing/feature_extraction_spacy.py

# 3. Proceed with analysis pipeline (steps above)
```

## System Requirements

### For Analysis Pipeline (Using Processed Data)

**Minimum:**
- Python 3.9+
- 16 GB RAM
- 4 CPU cores
- 10 GB disk space

**Recommended:**
- Python 3.10+
- 32 GB RAM
- 8 CPU cores
- 50 GB disk space

### For Full Preprocessing Pipeline

**Required:**
- Python 3.9+
- 64 GB RAM
- NVIDIA GPU with 16+ GB VRAM (CUDA 11.8+)
- 500 GB disk space
- High-speed internet for model downloads

## Key Dependencies

- **NLP & Transformers:**
  - `sentence-transformers>=2.2.0` (multilingual sentence embeddings)
  - `transformers>=4.30.0` (LLaMA models)
  - `spacy>=3.6.0` (syntactic analysis)

- **Machine Learning:**
  - `xgboost>=1.7.6` (gradient boosting)
  - `scikit-learn>=1.3.0` (ML utilities)
  - `shap>=0.42.0` (model interpretation)

- **Data Processing:**
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`

- **Visualization:**
  - `matplotlib>=3.7.0`
  - `seaborn>=0.12.0`

See `requirements.txt` for complete list with exact versions.

## Key Features

### Linguistic Complexity Measurement

**Five Independent Dimensions:**
1. **Lexical Complexity:** Vocabulary sophistication, word rarity
2. **Syntactic Complexity:** Sentence structure, parse tree depth
3. **Semantic Density:** Information content, propositional density
4. **Cohesion:** Referential links, discourse connectivity
5. **Cognitive Processing:** Perplexity, surprisal (from LLaMA 3.1)

**Nine Composite Indices:**
- Obfuscation Index
- Processing Difficulty Score
- Evasiveness Index
- Coherence Index
- Information Density Score
- Boilerplate Index
- Cognitive Load Index
- Accessibility Index
- Deception Likelihood Indicator

### Transformer-Based Feature Extraction

- **Sentence-BERT** (`paraphrase-multilingual-mpnet-base-v2`): Semantic coherence, similarity metrics
- **LLaMA 3.1-8B-Instruct:** Perplexity and surprisal for cognitive processing difficulty
- **SpaCy (nl_core_news_lg, fr_core_news_lg):** Syntactic features, dependency parsing

### Predictive Modeling

- **XGBoost** gradient boosting with hyperparameter optimization
- **SHAP** analysis for feature importance and interpretation
- **Temporal validation:** Rolling window approach
- **Class imbalance handling:** Custom loss functions (823:1 ratio)

## Results Overview

**Key findings from our paper:**

- **Linguistic features improve bankruptcy prediction:**
  - AUC-PR: 0.002 → 0.399 at 1-year horizon
  - AUC-PR: 0.002 → 0.236 at 2-year horizon

- **Five independent linguistic dimensions identified**
  - Factor analysis confirms multidimensionality
  - Regime-based clustering (formulaic vs. narrative modes)

- **Cross-language patterns:**
  - Dutch auditor reports: 71.9% high-complexity-high-coherence
  - French management reports: 45.9% high-complexity-low-coherence (paradoxical simplicity)

- **Temporal dynamics:**
  - Structural break in 2021-2022
  - Complexity declined, coherence remained stable
  - Coincides with COVID-19 and CSRD anticipation

## Citation

If you use this code or data in your research, please cite:

### BibTeX
```bibtex
@article{yourname2025linguistic,
  title={Linguistic Complexity and Financial Distress: Evidence from Transformer-Based Analysis of Belgian Firms},
  author={Your Name and Co-Author Name},
  journal={Accounting and Business Research},
  year={2025},
  publisher={Taylor \& Francis}
}
```

### Data Citation
```
Your Name. (2025). Processed Linguistic Features for Belgian Corporate 
Annual Reports 2019-2023 [Data set]. Zenodo. 
https://doi.org/10.5281/zenodo.XXXXXXX
```

### Code Citation
```
Your Name. (2025). Replication Package for "Linguistic Complexity and 
Financial Distress" [Software]. Zenodo. 
https://doi.org/10.5281/zenodo.YYYYYYY
```

## License

### Code
This repository's code is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Data
The processed linguistic features dataset is released under **CC0 1.0 Universal** (Public Domain Dedication) - see [LICENSE_DATA](LICENSE_DATA) file.

While not legally required, we kindly request that users cite our work (see Citation section above).

### Raw Data
Raw XBRL filings remain subject to original licensing terms from National Bank of Belgium and Bel-First database.

## Contributing

We welcome contributions! Please feel free to:
- Report bugs or issues
- Suggest enhancements
- Submit pull requests
- Ask questions

Please open an issue on GitHub or contact us directly.

## Documentation

Detailed documentation available in `docs/`:

- **[Data Dictionary](docs/data_dictionary.md):** Complete list of variables and definitions
- **[Methodology](docs/methodology.md):** Detailed explanation of methods
- **[Replication Guide](docs/replication_guide.md):** Step-by-step replication instructions

## Support

### Questions about the paper?
Contact: [your.email@institution.edu](mailto:your.email@institution.edu)

### Technical issues with code?
Open an issue: [GitHub Issues](https://github.com/yourusername/repo/issues)

### Commercial licensing inquiries?
Contact: [your.email@institution.edu](mailto:your.email@institution.edu)

## Acknowledgments

We thank participants at [conference/seminar names] for helpful comments. We are grateful to [data providers, research assistants, etc.] for assistance. All remaining errors are our own.

## Version History

- **v1.0.0** (2025-XX-XX): Initial release
  - Complete replication package
  - Processed data available on Zenodo
  - All analysis code

## Related Projects

- [Other related work if applicable]

---
## Screenshots
![Figure 1: Model Performance](outputs/figures/model_performance.png)
*Temporal evolution of model performance across prediction horizons*

## FAQ
**Q: Can I use this for commercial purposes?**
A: Yes, code is MIT licensed. Please cite our work.

**Q: How long does preprocessing take?**
A: ~200 hours on GPU infrastructure. We recommend using our processed data.
---
**Last Updated:** [Date]  
**Maintained by:** [Your Name]  
**Institution:** [Your Institution]
