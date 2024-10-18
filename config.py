from  pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 10**-4,
        "sql_len": 300,
        "d_model": 512,
        "datasource": 'Helsinki-NLP/opus_books',
        "srs_lang": "en",
        "tgt_lang": "es",
        "model_folder": "weights",
        "model_basename": "transformer_model_",
        "preload":    None,#"latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/transformer_model"
    }

'''
def save_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

'''
def save_weights_file_path(config, epoch: str):
    model_folder = Path(f"{config['datasource']}_{config['model_folder']}")
    model_filename = model_folder / f"{config['model_basename']}{epoch}.pt"

    # Create the model_folder if it doesn't exist
    model_folder.mkdir(parents=True, exist_ok=True)

    return str(model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])