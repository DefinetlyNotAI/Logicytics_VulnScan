# VulnScan Documentation

VulnScan is designed to detect sensitive data across various file formats.
It offers a modular framework to train models using diverse algorithms,
from traditional ML classifiers to advanced Neural Networks.

This document outlines the system's naming conventions, lifecycle, and model configuration.

> [!NOTE]
> Ported in update 3.5.0 of Logicytics - Latest update from there was 3.4.2
>
> You can find the main repo and generated files [here](https://github.com/DefinetlyNotAI/Logicytics/tree/main/CODE/vulnscan)

> [!IMPORTANT]
> Old documentation is available in the `Archived Models` directory of this [repository](https://github.com/DefinetlyNotAI/VulnScan_Data)
>
> This documentation covers test data, metrics and niche features.

---

## Naming Conventions

### Model Naming Format
`Model_{Type of model}.{Version}`

- **Type of Model**: Describes the training data configuration.
    - `SenseNano`: Test set with <10k files or <1k vals (PT), used for error-checking.
    - `SenseMini`: Dataset with 10k to 50k files or 1k-5k vals (PT). `Balanced size for effective training and resource efficiency`.
    - `Sense`: Sensitive data set with 50k to 100k files or 5k-10k (PT).
    - `SenseMacro`: Large dataset with >100k files or >10k (PT).

- **Version Format**: `{Version#}{c}{Repeat#}`
    - **Version#**: Increment for major code updates.
    - **c**: Model identifier (e.g., NeuralNetwork, BERT, etc.). See below for codes.
    - **Repeat#**: Number of times the same model was trained without significant code changes, used to improve consistency.
    - **-F**: Denotes a failed model or a corrupted model.

### Model Identifiers

| Code | Model Type                |
|------|---------------------------|
| `b`  | BERT                      |
| `dt` | DecisionTree              |
| `et` | ExtraTrees                |
| `g`  | GBM                       |
| `l`  | LSTM                      |
| `n`  | NeuralNetwork (preferred) |
| `nb` | NaiveBayes                |
| `r`  | RandomForestClassifier    |
| `lr` | Logistic Regression       |
| `v`  | SupportVectorMachine      |
| `x`  | XGBoost                   |

### Example
`Model Sense .1n2`:
- Dataset: `Sense` (50k files, 50KB each).
- Version: 1 (first major version).
- Model: `NeuralNetwork` (`n`).
- Repeat Count: 2 (second training run with no major code changes).

---

## Life Cycle Phases

### Version 1 (Deprecated)
- **Removed**: Small and weak codebase, replaced by `v3`.

1. Generate data.
2. Index paths.
3. Read paths.
4. Train models and iterate through epochs.
5. Produce outputs: data, graphs, and `.pkl` files.

---

### Version 2 (Deprecated)
- **Deprecation Reason**: Outdated methods for splitting and vectorizing data.

1. Load Data.
2. Split Data.
3. Vectorize Text.
4. Initialize Model.
5. Train Model.
6. Evaluate Model.
7. Save Model.
8. Track Progress.

---

### Version 3 (Superseded)
- **Superseded by Version 4**
- Retained for reference and backward compatibility.

1. **Read Config**: Load model and training parameters.
2. **Load Data**: Collect and preprocess sensitive data.
3. **Split Data**: Separate into training and validation sets.
4. **Vectorize Text**: Transform textual data using `TfidfVectorizer`.
5. **Initialize Model**: Define traditional ML or Neural Network models.
6. **Train Model**: Perform iterative training using epochs.
7. **Validate Model**: Evaluate with metrics and generate classification reports.
8. **Save Model**: Persist trained models and vectorizers for reuse.
9. **Track Progress**: Log and visualize accuracy and loss trends over epochs.

---

### Version 4 (Current)
- **Current Release**: Major improvements in scalability, modularity, and embedding-based training.
- **Key Features**:
    - **Dynamic Dataset Generation**: Uses GPT-Neo for synthetic sensitive data generation, scaling from small to large datasets.
    - **Embedding-Based Training**: Employs MiniLM sentence embeddings for all text samples, improving feature representation.
    - **Multi-Round Training**: Supports multiple training rounds per dataset size for robust model evaluation.
    - **Automated Caching**: Datasets and embeddings are cached for reuse, reducing redundant computation.
    - **Configurable Model Naming**: Model names reflect dataset size, type, version, and training round.
    - **Progress Tracking**: Training history and metrics are saved per round for analysis.
    - **Extensible Framework**: Easily integrates new models, datasets, and training strategies.

#### Version 4 Workflow
1. **Initialize Resources**: Load GPT-Neo and MiniLM models for generation and embedding.
2. **Dataset Generation**: Create or load datasets of varying sizes, using cached data when available.
3. **Embedding Generation**: Compute sentence embeddings for train, validation, and test splits.
4. **Split Data**: Partition data into train, validation, and test sets based on configurable ratios.
5. **Model Training**: Train a neural network using embeddings, with support for early stopping and learning rate scheduling.
6. **Multi-Round Evaluation**: Repeat training for each dataset size and round, saving metrics and model states.
7. **Progress Logging**: Save training history, plots, and logs for each round and model.
8. **Extensibility**: Easily add new dataset sizes, model types, or embedding strategies.

#### Example Model Name
`Model_Sense.4n1`:
- Dataset: `Sense` (50k to 100k files).
- Version: 4 (current major version).
- Model: NeuralNetwork (`n`).
- Training Round: 1.

---

## Preferred Model
**NeuralNetwork (`n`)**
- Proven to be the most effective for detecting sensitive data in the project.

---

## Notes
- **Naming System**: Helps track model versions, datasets, and training iterations for transparency and reproducibility.
- **Current Focus**: Version 4 for improved scalability, embedding-based training, and robust performance.

---

## Additional Features

- **Progress Tracking**: Visualizes accuracy and loss per epoch with graphs.
- **Error Handling**: Logs errors for missing files, attribute issues, or unexpected conditions.
- **Extensibility**: Supports plug-and-play integration for new algorithms or datasets.


# More files

There is a repository that archived all the data used to make the model,
as well as previously trained models for you to test out
(loading scripts and vectorizers are not included).

The repository is located [here](https://github.com/DefinetlyNotAI/VulnScan_Data).

The repository contains the following directories:
- `cache`: Contains all training data generated by [`Generator.py`](Generator.py).
- `NN features`: Contains information about the model `.3n3` and the vectorizer used. Information include:
    - `Documentation_Study_Network.md`: A markdown file that contains more info.
    - `Neural_Network_Nodes_Graph.gexf`: A Gephi file that contains the model nodes and edges.
    - `Feature_Importance.svg`: A SVG file that contains the feature importance of the model.
    - `Loss_Landscape_3D.html`: A HTML file that contains the 3D loss landscape of the model.
    - `Model_State_Dict.txt`: A text file that contains the model state dictionary.
    - `Model_Summary.txt`: A text file that contains the model summary.
    - `Model_Visualization.png`: A PNG file that contains the model visualization.
    - `Visualize_Activation.png`: A PNG file that contains the visualization of the model activation.
    - `Visualize_tSNE.png`: A PNG file that contains the visualization of the model t-SNE with the default training test embeds.
    - `Visualize_tSNE_custom.png`: A PNG file that contains the visualization of the model t-SNE with real world training examples (only 100).
    - `Weight_Distribution.png`: A PNG file that contains the weight distribution of the model.

