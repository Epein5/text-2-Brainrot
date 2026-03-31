
## PLANNED FOLDER STRUCTURE
```
brainrot-engine/
│
├── data/
│   ├── raw/                # Collected raw sentences
│   ├── synthetic/          # GPT-generated data
│   └── processed/          # Cleaned & formatted data ready for training
│
├── src/
│   ├── preprocessing/      # Scripts to clean & format text
│   │   ├── clean.py
│   │   ├── format.py
│   │   └── generate_data.py   # Script to generate synthetic dataset
│   │
│   ├── training/           # Fine-tuning logic
│   │   ├── train.py
│   │   ├── trainer.py
│   │   └── config.py
│   │
│   ├── evaluation/         # Scripts to evaluate outputs
│   │   └── eval.py
│   │
│   ├── inference/          # Scripts to generate brainrot text
│   │   ├── generate.py
│   │   └── pipeline.py
│   │
│   ├── utils/              # Helper functions used across scripts
│   │   └── helpers.py
│
├── models/                 # Saved checkpoints / LoRA weights
│
├── notebooks/              # Jupyter notebooks for experiments & testing
│
├── api/                    # FastAPI or Flask server
│   └── main.py
│
├── configs/                # YAML/JSON configs for training & inference
│   └── train.yaml
│
├── scripts/                # Helper scripts
│   ├── run_training.sh
│   └── run_inference.sh
│
├── requirements.txt        # Python dependencies
├── README.md               # Project description, badges, instructions
└── .env                    # API keys / environment variables
```
