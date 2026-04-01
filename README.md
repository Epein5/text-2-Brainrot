
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



## STEP 1: DATA COLLECTION

**How to collect data:**  
Just run `src/preprocessing/generate_data.py` to whip up some synthetic data.

It uses the Gemini model (we went with Gemini 3.1 Flash Lite since we had credits). 

Heads up: Gemini might cost money, so feel free to swap in a cheaper model if you want.

**Customizing data generation:**  
You can adjust the parameters in `src/preprocessing/generate_data.py` to control how much data is generated and which fields are included in the output. 

To generate different types of data, simply modify the system prompt in the script. 

The generated data will be saved to `data/synthetic/data.csv`.


