# MIAAM V2.0 evaluations

Knowledge Tracing experiments on the MIAAM dataset using [pyKT](https://github.com/pykt-team/pykt-toolkit).

The experiments reported in the paper use [MIAAM V2.0](https://huggingface.co/datasets/GAIMHE/MIAAM-V2/tree/v2.0). Use this version to reproduce the reported results.

The current dataset release is [MIAAM V2.1](https://huggingface.co/datasets/GAIMHE/MIAAM-V2). It includes subsequent deduplication and filtering updates and therefore differs from the data used in these experiments.

## PyKT

### 1. Process the dataset

Run `notebooks/pykt_processing.ipynb` until *"4) Construct train/test dataset for other baselines"*. The notebook first filters out exercises that have no screenshot in `data/screenshots/compressed/`, so all retained exercises have visual content available. It then produces two files in `miaam_pykt_data`:
- `data.txt` — student sequences in pyKT's format
- `gkt_graph_transition.npz` — activity-level adjacency matrix for graph-aware models (e.g. GKT), indexed by `activity_id_int`

### 2. Install pyKT

Get pykt-toolkit using `git clone https://github.com/pykt-team/pykt-toolkit.git`. Install it with:

```bash
pip install -e pykt-toolkit
```

### 3. Add MIAAM to pyKT

Copy the data and preprocessing script into pykt-toolkit:

```bash
mkdir pykt-toolkit/data/miaamv1
cp miaam_pykt_data/* pykt-toolkit/data/miaamv1
cp miaam_pykt_data/miaamv1_preprocess.py pykt-toolkit/pykt/preprocess/
```

Register the dataset path in `pykt-toolkit/examples/data_preprocess.py`:

```python
dname2paths = {
    "miaamv1": "../data/miaamv1/",
    ...
}
```

Register the preprocessing script in `pykt-toolkit/pykt/preprocess/data_preprocess.py`:

```python
elif dataset_name == "miaamv1":
    from .miaamv1_preprocess import read_data_from_csv
```

### 4. Run pyKT's data preprocessing

```bash
cd pykt-toolkit/examples
python data_preprocess.py --dataset_name=miaamv1
```

### 5. Break down the original dataset into train/test following PyKT's split
Run the section *"4) Construct train/test dataset for other baselines"* from `notebooks/pykt_processing.ipynb`.
This reads the student IDs from pyKT's split files and filters the enriched `maths_data_filtered.parquet` dataframe accordingly, producing `interactions_train.parquet` and `interactions_test.parquet` at the project root. Both files include the integer-encoded columns (`user_id_int`, `exercise_id_int`, `activity_id_int`) added in section 2, making them directly comparable with pyKT's results.

## LLM / VLM evaluation

Zero-shot knowledge tracing with an OpenRouter-served LLM (open or
closed-weight). Two tasks are supported:

- **prob** — predict the probability the next attempt is correct (compared
  against pyKT's AUC / accuracy / Brier).
- **answer** — on multiple-choice items, predict the option index the
  student will pick (top-1 accuracy against random and "always-correct"
  baselines).

See `evaluation_llm/README.md` for the full pipeline (build windows → run
eval → score), modality / expert-knowledge knobs, and SLURM submission.
Quick start once `interactions_test.parquet` and the dataset are in place:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
jupyter notebook notebooks/run_eval_openrouter.ipynb
```

## Licensing

The evaluation scripts in this repository are released under the [MIT License](https://opensource.org/licenses/MIT).

Running these scripts requires access to the [MIAAM dataset on Hugging Face](https://huggingface.co/datasets/GAIMHE/MIAAM-V2).

The current release use the [MIAAM Research Data Licence](https://huggingface.co/datasets/GAIMHE/MIAAM-V2/blob/main/LICENSE). Dataset usage is governed by the applicable terms for the version obtained.
