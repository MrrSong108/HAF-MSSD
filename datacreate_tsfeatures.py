import pdb
import time
from pathlib import Path
import pandas as pd
from tsfeatures import tsfeatures
from datetime import datetime, timedelta



def load_and_process_data(df, date):
    # Adjust to the tsfeatures format
    df["unique_id"] = pd.to_datetime(date)  # Use each day as an independent ID
    df.rename(columns={"time": "ds", "count": "y"}, inplace=True)

    return df


if __name__ == '__main__':
    input_dir = Path(r"Data")
    output_dir = Path(r"Data")
    start_date = datetime(2024, 5, 20)
    end_date = datetime(2024, 8, 24)
    stemp_date = start_date
    while stemp_date <= end_date:
        data = pd.DataFrame()
        for i in range(1, -1, -1):
            file_path = input_dir / f"{(stemp_date - timedelta(days=i)).strftime('%Y%m%d')}.csv"
            stemp_data = pd.read_csv(file_path)
            data = pd.concat([data, stemp_data], ignore_index=True)
        stemp_time = stemp_date
        data['time'] = pd.to_datetime(data['time'])
        all_features = pd.DataFrame()
        while stemp_time <= (stemp_date + timedelta(days=1) - timedelta(minutes=1)):
            count_time = time.time()
            df = data[(stemp_time - timedelta(days=1) <= data['time']) & (data['time'] < stemp_time)]
            # Calculate tsfeatures
            features = tsfeatures(load_and_process_data(df, stemp_date), freq=1440)
            timeseries = features.drop(columns=['unique_id'])
            timeseries['time'] = stemp_time.strftime('%Y-%m-%d %H:%M:%S')
            timeseries.insert(0, 'time', timeseries.pop('time'))
            all_features = pd.concat([all_features, timeseries], ignore_index=True)
            if len(all_features) % 60 == 0:
                print(f"Completed up to: {stemp_time.strftime('%Y-%m-%d %H:%M:%S')}")
            stemp_time +=timedelta(minutes=1)

        save_path = output_dir / f"{stemp_date.strftime('%Y%m%d')}.csv"
        all_features.to_csv(save_path, index=False)
        stemp_date += timedelta(days=1)