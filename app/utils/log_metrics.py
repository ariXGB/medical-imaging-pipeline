from pathlib import Path
import pandas as pd

def log_metrics(metrics, model_type):

    csv_path = Path(f"metrics/{model_type}_metrics.csv")

    new_row = pd.DataFrame([metrics])

    if csv_path.exists():

        existing_df = pd.read_csv(csv_path)

        df = pd.concat([existing_df, new_row],ignore_index=True)

    else:
        df = new_row

    csv_path.parent.mkdir(parents=True,exist_ok=True)

    df.to_csv(csv_path,index=False)