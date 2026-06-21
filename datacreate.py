import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message="DataFrame.mean and DataFrame.median with numeric_only=None"
)

# =========================
# 1. Basic path configuration
# =========================
# Please change RAW_DATA_ROOT to your own raw data directory.
# Do not upload the real raw data path to GitHub.

RAW_DATA_ROOT = Path(os.getenv(
    "RAW_DATA_ROOT",
    r"airport_raw_data"
))

OUTPUT_DIR = Path(os.getenv(
    "OUTPUT_DIR",
    r"airport_processed_data"
))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
start_date = datetime(2024, 7, 23)
end_date = datetime(2024,7, 24)
stemp_date = start_date
addminutes = timedelta(minutes=30)
addhour = timedelta(hours=23)

current_date = start_date

def get_file_path(current_date):
    """
    Return the raw data file paths for a given date.

    The raw data are organized by source:
    - queue: usually one table per day
    - trafft2 / trafft3: usually one table every few days
    - check: usually one table every few days
    - plane: usually one table every several days

    Please modify the relative paths according to your own data folder structure.
    """

    date_str = current_date.strftime("%Y%m%d")

    # =========================
    # 1. Queue data path
    # =========================
    if datetime(2024, 5, 15) <= current_date < datetime(2024, 7, 2):
        queue_path = (
            RAW_DATA_ROOT
            / "0515-0701"
            / "queue"
            / date_str
            / "queues.xlsx"
        )

    elif datetime(2024, 7, 2) <= current_date < datetime(2024, 8, 15):
        queue_path = (
            RAW_DATA_ROOT
            / "0702-0815"
            / "queue"
            / date_str
            / "queues.xlsx"
        )

    else:
        print("Date is outside the queue data range.")
        return None, None, None, None, None


    # =========================
    # 2. T2 terminal traffic data path
    # =========================
    if datetime(2024, 5, 15) <= current_date < datetime(2024, 5, 18):
        trafft2_path = (
            RAW_DATA_ROOT
            / "0515-0701"
            / "trafft2"
            / "515-518.xlsx"
        )

    elif datetime(2024, 5, 18) <= current_date < datetime(2024, 5, 21):
        trafft2_path = (
            RAW_DATA_ROOT
            / "0515-0701"
            / "trafft2"
            / "518-521.xlsx"
        )

    elif datetime(2024, 7, 23) <= current_date < datetime(2024, 7, 26):
        trafft2_path = (
            RAW_DATA_ROOT
            / "0702-0815"
            / "trafft2"
            / "726.xlsx"
        )

    elif datetime(2024, 7, 26) <= current_date < datetime(2024, 7, 29):
        trafft2_path = (
            RAW_DATA_ROOT
            / "0702-0815"
            / "trafft2"
            / "729.xlsx"
        )

    else:
        print("Date is outside the T2 traffic data range.")
        return None, None, None, None, None


    # =========================
    # 3. T3 terminal traffic data path
    # =========================
    if datetime(2024, 5, 15) <= current_date < datetime(2024, 5, 18):
        trafft3_path = (
            RAW_DATA_ROOT
            / "0515-0701"
            / "trafft3"
            / "0515+.xlsx"
        )

    elif datetime(2024, 5, 18) <= current_date < datetime(2024, 5, 21):
        trafft3_path = (
            RAW_DATA_ROOT
            / "0515-0701"
            / "trafft3"
            / "0518+.xlsx"
        )

    elif datetime(2024, 7, 23) <= current_date < datetime(2024, 7, 26):
        trafft3_path = (
            RAW_DATA_ROOT
            / "0702-0815"
            / "trafft3"
            / "726.xlsx"
        )

    elif datetime(2024, 7, 26) <= current_date < datetime(2024, 7, 29):
        trafft3_path = (
            RAW_DATA_ROOT
            / "0702-0815"
            / "trafft3"
            / "729.xlsx"
        )

    else:
        print("Date is outside the T3 traffic data range.")
        return None, None, None, None, None


    # =========================
    # 4. Security check data path
    # =========================
    if datetime(2024, 5, 15) <= current_date < datetime(2024, 5, 30):
        check_path = (
            RAW_DATA_ROOT
            / "0515-0701"
            / "check"
            / "515-530.xlsx"
        )

    elif datetime(2024, 5, 30) <= current_date < datetime(2024, 6, 15):
        check_path = (
            RAW_DATA_ROOT
            / "0515-0701"
            / "check"
            / "530-615.xlsx"
        )

    elif datetime(2024, 7, 20) <= current_date < datetime(2024, 7, 26):
        check_path = (
            RAW_DATA_ROOT
            / "0702-0815"
            / "check"
            / "720-725.xlsx"
        )

    elif datetime(2024, 7, 26) <= current_date < datetime(2024, 8, 1):
        check_path = (
            RAW_DATA_ROOT
            / "0702-0815"
            / "check"
            / "726-731.xlsx"
        )

    else:
        print("Date is outside the security check data range.")
        return None, None, None, None, None


    # =========================
    # 5. Flight data path
    # =========================
    if datetime(2024, 5, 15) <= current_date < datetime(2024, 5, 22):
        plane_path = (
            RAW_DATA_ROOT
            / "0515-0701"
            / "flight"
            / "515-522.xlsx"
        )

    elif datetime(2024, 5, 22) <= current_date < datetime(2024, 5, 29):
        plane_path = (
            RAW_DATA_ROOT
            / "0515-0701"
            / "flight"
            / "522-529.xlsx"
        )

    elif datetime(2024, 7, 17) <= current_date < datetime(2024, 8, 1):
        plane_path = (
            RAW_DATA_ROOT
            / "0702-0815"
            / "flight"
            / "716-730.xlsx"
        )

    elif datetime(2024, 8, 1) <= current_date < datetime(2024, 8, 19):
        plane_path = (
            RAW_DATA_ROOT
            / "0702-0815"
            / "flight"
            / "730-818.xlsx"
        )

    else:
        print("Date is outside the flight data range.")
        return None, None, None, None, None

    return queue_path, trafft2_path, trafft3_path, check_path, plane_path

while stemp_date <= end_date:
    count_time = time.time()
    source_customer = pd.DataFrame(columns=['start_time',

                                        'queue_countpassed5', 'queue_countpassed10',  'queue_countpassed15', 'queue_countpassed20',
                                        'queue_countpassed25', 'queue_countpassed30', 'queue_countpassed35', 'queue_countpassed40',
                                        'queue_countpassed60', 'queue_countpassed90', 'queue_countpassed120', 'queue_countpassed3h',
                                        'queue_countpassed4h', 'queue_countpassed5h', 'queue_countpassed6h', 'queue_countpassed7h',
                                        'queue_countpassed8h', 'queue_countpassed9h', 'queue_countpassed10h', 'queue_countpassed11h',
                                        'queue_countpassed12h','queue_countpassed13h', 'queue_countpassed14h', 'queue_countpassed15h',
                                        'queue_countpassed16h','queue_countpassed17h', 'queue_countpassed18h', 'queue_countpassed19h',
                                        'queue_countpassed20h', 'queue_countpassed21h', 'queue_countpassed22h', 'queue_countpassed23h',
                                        'queue_countpassed24h',

                                        'queue_maxpassed5', 'queue_maxpassed10',  'queue_maxpassed15', 'queue_maxpassed20',
                                        'queue_maxpassed25', 'queue_maxpassed30', 'queue_maxpassed35', 'queue_maxpassed40',
                                        'queue_maxpassed60', 'queue_maxpassed90',  'queue_maxpassed120', 'queue_maxpassed3h',
                                        'queue_maxpassed4h', 'queue_maxpassed5h', 'queue_maxpassed6h', 'queue_maxpassed7h',
                                        'queue_maxpassed8h','queue_maxpassed9h', 'queue_maxpassed10h', 'queue_maxpassed11h',
                                        'queue_maxpassed12h', 'queue_maxpassed13h', 'queue_maxpassed14h', 'queue_maxpassed15h',
                                        'queue_maxpassed16h','queue_maxpassed17h', 'queue_maxpassed18h', 'queue_maxpassed19h',
                                        'queue_maxpassed20h','queue_maxpassed21h', 'queue_maxpassed22h', 'queue_maxpassed23h',
                                        'queue_maxpassed24h',

                                        'queue_minpassed5', 'queue_minpassed10', 'queue_minpassed15', 'queue_minpassed20',
                                        'queue_minpassed25', 'queue_minpassed30', 'queue_minpassed35', 'queue_minpassed40',
                                        'queue_minpassed60', 'queue_minpassed90', 'queue_minpassed120', 'queue_minpassed3h',
                                        'queue_minpassed4h', 'queue_minpassed5h', 'queue_minpassed6h', 'queue_minpassed7h',
                                        'queue_minpassed8h', 'queue_minpassed9h', 'queue_minpassed10h', 'queue_minpassed11h',
                                        'queue_minpassed12h', 'queue_minpassed13h', 'queue_minpassed14h', 'queue_minpassed15h',
                                        'queue_minpassed16h', 'queue_minpassed17h', 'queue_minpassed18h', 'queue_minpassed19h',
                                        'queue_minpassed20h', 'queue_minpassed21h', 'queue_minpassed22h', 'queue_minpassed23h',
                                        'queue_minpassed24h',

                                        'count_passed5', 'count_passed10', 'count_passed15', 'count_passed20',
                                        'count_passed25', 'count_passed30', 'count_passed35', 'count_passed40',
                                        'count_passed60', 'count_passed90', 'count_passed120', 'count_passed3h',
                                        'count_passed4h', 'count_passed5h', 'count_passed6h', 'count_passed7h',
                                        'count_passed8h', 'count_passed9h', 'count_passed10h', 'count_passed11h',
                                        'count_passed12h', 'count_passed13h', 'count_passed14h', 'count_passed15h',
                                        'count_passed16h', 'count_passed17h', 'count_passed18h', 'count_passed19h',
                                        'count_passed20h', 'count_passed21h', 'count_passed22h', 'count_passed23h',
                                        'count_passed24h',

                                        'wait_passed5', 'wait_passed10', 'wait_passed15', 'wait_passed20',
                                        'wait_passed25', 'wait_passed30', 'wait_passed35', 'wait_passed40',

                                        'wait_max5', 'wait_max10', 'wait_max15', 'wait_max20',
                                        'wait_max25', 'wait_max30', 'wait_max35', 'wait_max40',
                                        'wait_max60', 'wait_max90', 'wait_max120', 'wait_max3h',
                                        'wait_max4h', 'wait_max5h', 'wait_max6h', 'wait_max7h',
                                        'wait_max8h', 'wait_max9h', 'wait_max10h', 'wait_max11h',
                                        'wait_max12h', 'wait_max13h', 'wait_max14h', 'wait_max15h',
                                        'wait_max16h', 'wait_max17h', 'wait_max18h', 'wait_max19h',
                                        'wait_max20h', 'wait_max21h', 'wait_max22h', 'wait_max23h',
                                        'wait_max24h',

                                        'wait_min5', 'wait_min10', 'wait_min15', 'wait_min20',
                                        'wait_min25', 'wait_min30', 'wait_min35', 'wait_min40',
                                        'wait_min60', 'wait_min90', 'wait_min120', 'wait_min3h',
                                        'wait_min4h', 'wait_min5h', 'wait_min6h', 'wait_min7h',
                                        'wait_min8h', 'wait_min9h', 'wait_min10h', 'wait_min11h',
                                        'wait_min12h', 'wait_min13h', 'wait_min14h', 'wait_min15h',
                                        'wait_min16h', 'wait_min17h', 'wait_min18h', 'wait_min19h',
                                        'wait_min20h', 'wait_min21h', 'wait_min22h', 'wait_min23h',
                                        'wait_min24h',

                                        'wait_avr5', 'wait_avr10', 'wait_avr15', 'wait_avr20',
                                        'wait_avr25', 'wait_avr30', 'wait_avr35', 'wait_avr40',
                                        'wait_avr60', 'wait_avr90', 'wait_avr120', 'wait_avr3h',
                                        'wait_avr4h', 'wait_avr5h', 'wait_avr6h', 'wait_avr7h',
                                        'wait_avr8h', 'wait_avr9h', 'wait_avr10h', 'wait_avr11h',
                                        'wait_avr12h', 'wait_avr13h', 'wait_avr14h', 'wait_avr15h',
                                        'wait_avr16h', 'wait_avr17h', 'wait_avr18h', 'wait_avr19h',
                                        'wait_avr20h', 'wait_avr21h', 'wait_avr22h', 'wait_avr23h',
                                        'wait_avr24h',

                                        'check_countpassed5', 'check_countpassed10', 'check_countpassed15', 'check_countpassed20',
                                        'check_countpassed25', 'check_countpassed30', 'check_countpassed35', 'check_countpassed40',
                                        'check_countpassed60', 'check_countpassed90', 'check_countpassed120', 'check_countpassed3h',
                                        'check_countpassed4h', 'check_countpassed5h', 'check_countpassed6h', 'check_countpassed7h',
                                        'check_countpassed8h', 'check_countpassed9h', 'check_countpassed10h', 'check_countpassed11h',
                                        'check_countpassed12h', 'check_countpassed13h', 'check_countpassed14h', 'check_countpassed15h',
                                        'check_countpassed16h', 'check_countpassed17h', 'check_countpassed18h', 'check_countpassed19h',
                                        'check_countpassed20h', 'check_countpassed21h', 'check_countpassed22h', 'check_countpassed23h',
                                        'check_countpassed24h',

                                        'check_T2passed5', 'check_T2passed10', 'check_T2passed15', 'check_T2passed20',
                                        'check_T2passed25', 'check_T2passed30', 'check_T2passed35', 'check_T2passed40',
                                        'check_T2passed60', 'check_T2passed90', 'check_T2passed120', 'check_T2passed3h',
                                        'check_T2passed4h', 'check_T2passed5h', 'check_T2passed6h', 'check_T2passed7h',
                                        'check_T2passed8h', 'check_T2passed9h', 'check_T2passed10h', 'check_T2passed11h',
                                        'check_T2passed12h', 'check_T2passed13h', 'check_T2passed14h', 'check_T2passed15h',
                                        'check_T2passed16h', 'check_T2passed17h', 'check_T2passed18h', 'check_T2passed19h',
                                        'check_T2passed20h', 'check_T2passed21h', 'check_T2passed22h', 'check_T2passed23h',
                                        'check_T2passed24h',

                                        'check_T3passed5', 'check_T3passed10', 'check_T3passed15', 'check_T3passed20',
                                        'check_T3passed25', 'check_T3passed30', 'check_T3passed35', 'check_T3passed40',
                                        'check_T3passed60', 'check_T3passed90', 'check_T3passed120', 'check_T3passed3h',
                                        'check_T3passed4h', 'check_T3passed5h', 'check_T3passed6h', 'check_T3passed7h',
                                        'check_T3passed8h', 'check_T3passed9h', 'check_T3passed10h', 'check_T3passed11h',
                                        'check_T3passed12h', 'check_T3passed13h', 'check_T3passed14h', 'check_T3passed15h',
                                        'check_T3passed16h', 'check_T3passed17h', 'check_T3passed18h', 'check_T3passed19h',
                                        'check_T3passed20h', 'check_T3passed21h', 'check_T3passed22h', 'check_T3passed23h',
                                        'check_T3passed24h',

                                        'check_manpassed5', 'check_manpassed10',  'check_manpassed15', 'check_manpassed20',
                                        'check_manpassed25', 'check_manpassed30', 'check_manpassed35', 'check_manpassed40',
                                        'check_manpassed60', 'check_manpassed90', 'check_manpassed120', 'check_manpassed3h',
                                        'check_manpassed4h', 'check_manpassed5h', 'check_manpassed6h', 'check_manpassed7h',
                                        'check_manpassed8h', 'check_manpassed9h', 'check_manpassed10h', 'check_manpassed11h',
                                        'check_manpassed12h', 'check_manpassed13h', 'check_manpassed14h', 'check_manpassed15h',
                                        'check_manpassed16h', 'check_manpassed17h', 'check_manpassed18h', 'check_manpassed19h',
                                        'check_manpassed20h', 'check_manpassed21h', 'check_manpassed22h', 'check_manpassed23h',
                                        'check_manpassed24h',

                                        'check_womanpassed5', 'check_womanpassed10', 'check_womanpassed15', 'check_womanpassed20',
                                        'check_womanpassed25', 'check_womanpassed30', 'check_womanpassed35', 'check_womanpassed40',
                                        'check_womanpassed60','check_womanpassed90', 'check_womanpassed120', 'check_womanpassed3h',
                                        'check_womanpassed4h', 'check_womanpassed5h', 'check_womanpassed6h', 'check_womanpassed7h',
                                        'check_womanpassed8h', 'check_womanpassed9h', 'check_womanpassed10h', 'check_womanpassed11h',
                                        'check_womanpassed12h', 'check_womanpassed13h', 'check_womanpassed14h', 'check_womanpassed15h',
                                        'check_womanpassed16h', 'check_womanpassed17h', 'check_womanpassed18h', 'check_womanpassed19h',
                                        'check_womanpassed20h', 'check_womanpassed21h', 'check_womanpassed22h', 'check_womanpassed23h',
                                        'check_womanpassed24h',

                                        'check_waitmax5', 'check_waitmax10', 'check_waitmax15', 'check_waitmax20',
                                        'check_waitmax25', 'check_waitmax30', 'check_waitmax35', 'check_waitmax40',
                                        'check_waitmax60', 'check_waitmax90', 'check_waitmax120', 'check_waitmax3h',
                                        'check_waitmax4h', 'check_waitmax5h', 'check_waitmax6h', 'check_waitmax7h',
                                        'check_waitmax8h', 'check_waitmax9h', 'check_waitmax10h', 'check_waitmax11h',
                                        'check_waitmax12h', 'check_waitmax13h', 'check_waitmax14h', 'check_waitmax15h',
                                        'check_waitmax16h', 'check_waitmax17h', 'check_waitmax18h', 'check_waitmax19h',
                                        'check_waitmax20h', 'check_waitmax21h', 'check_waitmax22h', 'check_waitmax23h',
                                        'check_waitmax24h',

                                        'check_waitmin5', 'check_waitmin10', 'check_waitmin15', 'check_waitmin20',
                                        'check_waitmin25', 'check_waitmin30', 'check_waitmin35', 'check_waitmin40',
                                        'check_waitmin60', 'check_waitmin90', 'check_waitmin120', 'check_waitmin3h',
                                        'check_waitmin4h', 'check_waitmin5h', 'check_waitmin6h', 'check_waitmin7h',
                                        'check_waitmin8h', 'check_waitmin9h', 'check_waitmin10h', 'check_waitmin11h',
                                        'check_waitmin12h', 'check_waitmin13h', 'check_waitmin14h', 'check_waitmin15h',
                                        'check_waitmin16h', 'check_waitmin17h', 'check_waitmin18h', 'check_waitmin19h',
                                        'check_waitmin20h', 'check_waitmin21h', 'check_waitmin22h', 'check_waitmin23h',
                                        'check_waitmin24h',

                                        'check_waitavr5', 'check_waitavr10', 'check_waitavr15', 'check_waitavr20',
                                        'check_waitavr25', 'check_waitavr30', 'check_waitavr35', 'check_waitavr40',
                                        'check_waitavr60', 'check_waitavr90', 'check_waitavr120', 'check_waitavr3h',
                                        'check_waitavr4h', 'check_waitavr5h', 'check_waitavr6h', 'check_waitavr7h',
                                        'check_waitavr8h', 'check_waitavr9h', 'check_waitavr10h', 'check_waitavr11h',
                                        'check_waitavr12h', 'check_waitavr13h', 'check_waitavr14h', 'check_waitavr15h',
                                        'check_waitavr16h', 'check_waitavr17h', 'check_waitavr18h', 'check_waitavr19h',
                                        'check_waitavr20h', 'check_waitavr21h', 'check_waitavr22h', 'check_waitavr23h',
                                        'check_waitavr24h',

                                        'traff_t2enterpassed5', 'traff_t2enterpassed10', 'traff_t2enterpassed15',
                                        'traff_t2enterpassed20', 'traff_t2enterpassed25', 'traff_t2enterpassed30',
                                        'traff_t2enterpassed35', 'traff_t2enterpassed40', 'traff_t2enterpassed60',
                                        'traff_t2enterpassed90', 'traff_t2enterpassed120', 'traff_t2enterpassed3h',
                                        'traff_t2enterpassed4h', 'traff_t2enterpassed5h', 'traff_t2enterpassed6h',
                                        'traff_t2enterpassed7h', 'traff_t2enterpassed8h', 'traff_t2enterpassed9h',
                                        'traff_t2enterpassed10h', 'traff_t2enterpassed11h', 'traff_t2enterpassed12h',
                                        'traff_t2enterpassed13h', 'traff_t2enterpassed14h', 'traff_t2enterpassed15h',
                                        'traff_t2enterpassed16h', 'traff_t2enterpassed17h', 'traff_t2enterpassed18h',
                                        'traff_t2enterpassed19h', 'traff_t2enterpassed20h', 'traff_t2enterpassed21h',
                                        'traff_t2enterpassed22h', 'traff_t2enterpassed23h', 'traff_t2enterpassed24h',

                                        'traff_t2exitpassed5', 'traff_t2exitpassed10', 'traff_t2exitpassed15',
                                        'traff_t2exitpassed20', 'traff_t2exitpassed25', 'traff_t2exitpassed30',
                                        'traff_t2exitpassed35', 'traff_t2exitpassed40', 'traff_t2exitpassed60',
                                        'traff_t2exitpassed90', 'traff_t2exitpassed120', 'traff_t2exitpassed3h',
                                        'traff_t2exitpassed4h', 'traff_t2exitpassed5h', 'traff_t2exitpassed6h',
                                        'traff_t2exitpassed7h', 'traff_t2exitpassed8h', 'traff_t2exitpassed9h',
                                        'traff_t2exitpassed10h', 'traff_t2exitpassed11h', 'traff_t2exitpassed12h',
                                        'traff_t2exitpassed13h', 'traff_t2exitpassed14h', 'traff_t2exitpassed15h',
                                        'traff_t2exitpassed16h', 'traff_t2exitpassed17h', 'traff_t2exitpassed18h',
                                        'traff_t2exitpassed19h', 'traff_t2exitpassed20h', 'traff_t2exitpassed21h',
                                        'traff_t2exitpassed22h', 'traff_t2exitpassed23h', 'traff_t2exitpassed24h',

                                        'traff_t3enterpassed5', 'traff_t3enterpassed10', 'traff_t3enterpassed15',
                                        'traff_t3enterpassed20', 'traff_t3enterpassed25', 'traff_t3enterpassed30',
                                        'traff_t3enterpassed35', 'traff_t3enterpassed40', 'traff_t3enterpassed60',
                                        'traff_t3enterpassed90', 'traff_t3enterpassed120', 'traff_t3enterpassed3h',
                                        'traff_t3enterpassed4h', 'traff_t3enterpassed5h', 'traff_t3enterpassed6h',
                                        'traff_t3enterpassed7h', 'traff_t3enterpassed8h', 'traff_t3enterpassed9h',
                                        'traff_t3enterpassed10h', 'traff_t3enterpassed11h', 'traff_t3enterpassed12h',
                                        'traff_t3enterpassed13h', 'traff_t3enterpassed14h', 'traff_t3enterpassed15h',
                                        'traff_t3enterpassed16h', 'traff_t3enterpassed17h', 'traff_t3enterpassed18h',
                                        'traff_t3enterpassed19h', 'traff_t3enterpassed20h', 'traff_t3enterpassed21h',
                                        'traff_t3enterpassed22h', 'traff_t3enterpassed23h', 'traff_t3enterpassed24h',

                                        'traff_t3exitpassed5', 'traff_t3exitpassed10', 'traff_t3exitpassed15',
                                        'traff_t3exitpassed20', 'traff_t3exitpassed25', 'traff_t3exitpassed30',
                                        'traff_t3exitpassed35', 'traff_t3exitpassed40', 'traff_t3exitpassed60',
                                        'traff_t3exitpassed90', 'traff_t3exitpassed120', 'traff_t3exitpassed3h',
                                        'traff_t3exitpassed4h', 'traff_t3exitpassed5h', 'traff_t3exitpassed6h',
                                        'traff_t3exitpassed7h', 'traff_t3exitpassed8h', 'traff_t3exitpassed9h',
                                        'traff_t3exitpassed10h', 'traff_t3exitpassed11h', 'traff_t3exitpassed12h',
                                        'traff_t3exitpassed13h', 'traff_t3exitpassed14h', 'traff_t3exitpassed15h',
                                        'traff_t3exitpassed16h', 'traff_t3exitpassed17h', 'traff_t3exitpassed18h',
                                        'traff_t3exitpassed19h', 'traff_t3exitpassed20h', 'traff_t3exitpassed21h',
                                        'traff_t3exitpassed22h', 'traff_t3exitpassed23h', 'traff_t3exitpassed24h',

                                        'plane_in30', 'plane_in60', 'plane_in90', 'plane_in120', 'plane_in3', 'plane_in4', 'plane_in5',
                                        'plane_in6', 'plane_in7', 'plane_in8', 'plane_in9', 'plane_in10', 'plane_in11', 'plane_in12',

                                        'plane_mix30', 'plane_mix60', 'plane_mix90', 'plane_mix120', 'plane_mix3', 'plane_mix4', 'plane_mix5',
                                        'plane_mix6', 'plane_mix7', 'plane_mix8', 'plane_mix9', 'plane_mix10', 'plane_mix11', 'plane_mix12',

                                        'plane_t2_30', 'plane_t2_60',  'plane_t2_90', 'plane_t2_120', 'plane_t2_3', 'plane_t2_4', 'plane_t2_5',
                                        'plane_t2_6', 'plane_t2_7', 'plane_t2_8', 'plane_t2_9', 'plane_t2_10', 'plane_t2_11',  'plane_t2_12',

                                        'plane_t3_30', 'plane_t3_60',  'plane_t3_90', 'plane_t3_120', 'plane_t3_3', 'plane_t3_4', 'plane_t3_5',
                                        'plane_t3_6', 'plane_t3_7', 'plane_t3_8', 'plane_t3_9', 'plane_t3_10', 'plane_t3_11',  'plane_t3_12',

                                        'plane_num_30','plane_num_60','plane_num_90','plane_num_120','plane_num_3','plane_num_4','plane_num_5',
                                        'plane_num_6','plane_num_7','plane_num_8','plane_num_9','plane_num_10','plane_num_11','plane_num_12',

                                        #预测目标
                                        'predict_T2_0.5', 'predict_T2_1', 'predict_T2_1.5', 'predict_T2_2', 'predict_T2_3',
                                        'predict_T2_4', 'predict_T2_5', 'predict_T2_6', 'predict_T2_7', 'predict_T2_8',
                                        'predict_T2_9', 'predict_T2_10', 'predict_T2_11', 'predict_T2_12', 'predict_T2_13',
                                        'predict_T2_14', 'predict_T2_15', 'predict_T2_16', 'predict_T2_17', 'predict_T2_18',
                                        'predict_T2_19', 'predict_T2_20', 'predict_T2_21', 'predict_T2_22', 'predict_T2_23', 'predict_T2_24',

                                        'predict_T3_0.5', 'predict_T3_1', 'predict_T3_1.5', 'predict_T3_2', 'predict_T3_3',
                                        'predict_T3_4', 'predict_T3_5', 'predict_T3_6', 'predict_T3_7', 'predict_T3_8',
                                        'predict_T3_9', 'predict_T3_10', 'predict_T3_11', 'predict_T3_12', 'predict_T3_13',
                                        'predict_T3_14', 'predict_T3_15', 'predict_T3_16', 'predict_T3_17', 'predict_T3_18',
                                        'predict_T3_19', 'predict_T3_20', 'predict_T3_21', 'predict_T3_22', 'predict_T3_23', 'predict_T3_24'
                                            ])
    queue = pd.DataFrame()
    trafft2 = pd.DataFrame()
    trafft3 = pd.DataFrame()
    check = pd.DataFrame()
    plane = pd.DataFrame()
    print("Building date:", stemp_date.strftime("%Y%m%d"), "\nLoading raw data...")
    if stemp_date >= (end_date - timedelta(days=1)):
        for i in range(3):
            queue_path, trafft2_path, trafft3_path, check_path, plane_path = get_file_path(stemp_date - timedelta(days=1) + timedelta(days=i))
            print((stemp_date - timedelta(days=1) + timedelta(days=i)).strftime('%Y%m%d'))
            if queue_path is None:
                print("Missing file path. Skip current date.")
                continue
            print(queue_path, '\n', trafft2_path, '\n', trafft3_path, '\n', check_path, '\n', plane_path)
            # Load Excel files and concatenate data.
            queue_df = pd.read_excel(queue_path)
            trafft2_df = pd.read_excel(trafft2_path)
            trafft3_df = pd.read_excel(trafft3_path)
            check_df = pd.read_excel(check_path)
            plane_df = pd.read_excel(plane_path)
            # Concatenate all raw data tables.
            queue = pd.concat([queue, queue_df], ignore_index=True)
            trafft2 = pd.concat([trafft2, trafft2_df], ignore_index=True)
            trafft3 = pd.concat([trafft3, trafft3_df], ignore_index=True)
            check = pd.concat([check, check_df], ignore_index=True)
            plane = pd.concat([plane, plane_df], ignore_index=True)
    else:
        for i in range(3):
            queue_path, trafft2_path, trafft3_path, check_path, plane_path = get_file_path(stemp_date - timedelta(days=1) + timedelta(days=i))
            print((stemp_date - timedelta(days=1) + timedelta(days=i)).strftime('%Y%m%d'))
            print(queue_path, '\n', trafft2_path, '\n', trafft3_path, '\n', check_path, '\n', plane_path)
            # Load Excel files and concatenate data.
            queue_df = pd.read_excel(queue_path)
            trafft2_df = pd.read_excel(trafft2_path)
            trafft3_df = pd.read_excel(trafft3_path)
            check_df = pd.read_excel(check_path)
            plane_df = pd.read_excel(plane_path)
            # Concatenate all raw data tables.
            queue = pd.concat([queue, queue_df])
            trafft2 = pd.concat([trafft2, trafft2_df])
            trafft3 = pd.concat([trafft3, trafft3_df])
            check = pd.concat([check, check_df])
            plane = pd.concat([plane, plane_df])
    print("All raw data loaded. Runtime:", time.time() - count_time)
    count_time = time.time()
    queue['date'] = pd.to_datetime(queue['date'])

    # Calculate the estimated queue arrival time.
    qd = queue.copy()
    qd['time'] = qd['date'] - pd.to_timedelta(qd['wait_time'], unit='s')

    # Process security check data time fields.
    cd = check.copy()
    cd['check_time'] = pd.to_datetime(cd['check_time'])
    cd['time'] = cd['check_time']
    cd = cd[cd['security_check_point'].str.contains('DC', na=False)]

    # Process terminal traffic data time fields.
    trafft2['time'] = pd.to_datetime(trafft2['time'])
    trafft3['time'] = pd.to_datetime(trafft3['time'])
    plane_data = plane.copy()
    stemp_time = datetime(2024,7,23,7,10)
    qd = qd.drop_duplicates().fillna(0)
    cd = cd.drop_duplicates().fillna(0)
    trafft2 = trafft2.drop_duplicates().fillna(0)
    trafft3 = trafft3.drop_duplicates().fillna(0)
    plane_data = plane_data.drop_duplicates().fillna(0)
    print("Available check data time range:", min(cd["time"]), "-", max(cd["time"]))
    while stemp_time <= (stemp_date + timedelta(days=1) - timedelta(minutes=1)):
        stemp_count = time.time()
        time_deltas = [
            ('5', timedelta(minutes=5)), ('10', timedelta(minutes=10)), ('15', timedelta(minutes=15)),
            ('20', timedelta(minutes=20)), ('25', timedelta(minutes=25)), ('30', timedelta(minutes=30)),
            ('35', timedelta(minutes=35)), ('40', timedelta(minutes=40)), ('60', timedelta(minutes=60)),
            ('90', timedelta(minutes=90)), ('120', timedelta(minutes=120))]
        stemp_queuedatapassed = {}
        stemp_checkdatas = {}
        stemp_trafft2datapassed = {}
        stemp_trafft3datapassed = {}
        stemp_checkwaits = {}
        stemp_planes = {}

        planes = {}
        source = {}
        stemp_waitpassed5 = qd[
            (stemp_time - timedelta(minutes=5) < qd['date']) & (qd['date'] < stemp_time - timedelta(minutes=4))]
        stemp_waitpassed10 = qd[
            (stemp_time - timedelta(minutes=10) < qd['date']) & (qd['date'] < stemp_time - timedelta(minutes=9))]
        stemp_waitpassed15 = qd[
            (stemp_time - timedelta(minutes=15) < qd['date']) & (qd['date'] < stemp_time - timedelta(minutes=14))]
        stemp_waitpassed20 = qd[
            (stemp_time - timedelta(minutes=20) < qd['date']) & (qd['date'] < stemp_time - timedelta(minutes=19))]
        stemp_waitpassed25 = qd[
            (stemp_time - timedelta(minutes=25) < qd['date']) & (qd['date'] < stemp_time - timedelta(minutes=24))]
        stemp_waitpassed30 = qd[
            (stemp_time - timedelta(minutes=30) < qd['date']) & (qd['date'] < stemp_time - timedelta(minutes=29))]
        stemp_waitpassed35 = qd[
            (stemp_time - timedelta(minutes=35) < qd['date']) & (qd['date'] < stemp_time - timedelta(minutes=34))]
        stemp_waitpassed40 = qd[
            (stemp_time - timedelta(minutes=40) < qd['date']) & (qd['date'] < stemp_time - timedelta(minutes=39))]

        # [T ~ T+30 min]
        stemp_countdata = qd[(stemp_time < qd['time']) & (qd['time'] < (stemp_time + addminutes))]

        # [T-n min ~ T]
        for i in range(1, 25):
            if i <= 2:
                for label, delta in time_deltas:
                    stemp_queuedatapassed[label] = qd[
                        (stemp_time - delta < qd['time']) & (qd['time'] < stemp_time)]
            else:
                stemp_queuedatapassed[f'{i}h'] = qd[(stemp_time - timedelta(hours=i) < qd['time']) &
                                                           (qd['time'] < stemp_time - timedelta(hours=i - 1))]

        for i in range(1, 25):
            if i <= 2:
                for label, delta in time_deltas:
                    stemp_checkdatas[label] = cd[
                        (stemp_time - delta < cd['time']) & (cd['time'] < stemp_time)]
            else:
                stemp_checkdatas[f'{i}h'] = cd[(stemp_time - timedelta(hours=i) < cd['time']) &
                                                    (cd['time'] < stemp_time - timedelta(hours=i - 1))]

        stemp_predict30 = cd[(stemp_time <= cd['time']) & (cd['time'] < stemp_time + timedelta(minutes=30))]
        stemp_predict60 = cd[(stemp_time + timedelta(minutes=30) <= cd['time']) & (cd['time'] < stemp_time + timedelta(minutes=60))]
        stemp_predict90 = cd[(stemp_time + timedelta(minutes=60) <= cd['time']) & (cd['time'] < stemp_time + timedelta(minutes=90))]
        stemp_predict120 = cd[(stemp_time + timedelta(minutes=90) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=2))]
        stemp_predict3 = cd[(stemp_time + timedelta(hours=2) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=3))]
        stemp_predict4 = cd[(stemp_time + timedelta(hours=3) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=4))]
        stemp_predict5 = cd[(stemp_time + timedelta(hours=4) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=5))]
        stemp_predict6 = cd[(stemp_time + timedelta(hours=5) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=6))]
        stemp_predict7 = cd[(stemp_time + timedelta(hours=6) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=7))]
        stemp_predict8 = cd[(stemp_time + timedelta(hours=7) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=8))]
        stemp_predict9 = cd[(stemp_time + timedelta(hours=8) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=9))]
        stemp_predict10 = cd[(stemp_time + timedelta(hours=9) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=10))]
        stemp_predict11 = cd[(stemp_time + timedelta(hours=10) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=11))]
        stemp_predict12 = cd[(stemp_time + timedelta(hours=11) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=12))]
        stemp_predict13 = cd[(stemp_time + timedelta(hours=12) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=13))]
        stemp_predict14 = cd[(stemp_time + timedelta(hours=13) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=14))]
        stemp_predict15 = cd[(stemp_time + timedelta(hours=14) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=15))]
        stemp_predict16 = cd[(stemp_time + timedelta(hours=15) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=16))]
        stemp_predict17 = cd[(stemp_time + timedelta(hours=16) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=17))]
        stemp_predict18 = cd[(stemp_time + timedelta(hours=17) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=18))]
        stemp_predict19 = cd[(stemp_time + timedelta(hours=18) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=19))]
        stemp_predict20 = cd[(stemp_time + timedelta(hours=19) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=20))]
        stemp_predict21 = cd[(stemp_time + timedelta(hours=20) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=21))]
        stemp_predict22 = cd[(stemp_time + timedelta(hours=21) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=22))]
        stemp_predict23 = cd[(stemp_time + timedelta(hours=22) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=23))]
        stemp_predict24 = cd[(stemp_time + timedelta(hours=23) <= cd['time']) & (cd['time'] < stemp_time + timedelta(hours=24))]

        for i in range(1, 25):
            if i <= 2:
                for label, delta in time_deltas:
                    stemp_trafft2datapassed[label] = trafft2[
                        (stemp_time - delta < trafft2['time']) & (trafft2['time'] < stemp_time)]
            else:
                stemp_trafft2datapassed[f'{i}h'] = trafft2[(stemp_time - timedelta(hours=i) < trafft2['time']) &
                                                           (trafft2['time'] < stemp_time - timedelta(hours=i - 1))]

        for i in range(1, 25):
            if i <= 2:
                for label, delta in time_deltas:
                    stemp_trafft3datapassed[label] = trafft3[
                        (stemp_time - delta < trafft3['time']) & (trafft3['time'] < stemp_time)]
            else:
                stemp_trafft3datapassed[f'{i}h'] = trafft3[(stemp_time - timedelta(hours=i) < trafft3['time']) &
                    (trafft3['time'] < stemp_time - timedelta(hours=i - 1))]

        plane_deltas = [
            ('30', timedelta(minutes=30)), ('60', timedelta(minutes=60)),
            ('90', timedelta(minutes=90)), ('120', timedelta(minutes=120))]
        time_intervals = [30, 60, 90, 120] + [i * 60 for i in range(2, 13)]
        for i in range(1, 13):
            if i <= 2:
                for label, delta in plane_deltas:
                    stemp_planes[f'stemp_plane{label}'] = plane_data[(stemp_time <= plane_data['selected_est_out_at']) &
                    (plane_data['selected_est_out_at'] < stemp_time+delta)]
            else:
                stemp_planes[f'stemp_plane{i}'] = plane_data[(stemp_time + timedelta(hours=i-1) <= plane_data['selected_est_out_at']) &
                                                 (plane_data['selected_est_out_at'] < stemp_time + timedelta(hours=i))]

        wait_avrs = {}
        for i in range(1, 25):
            if i <= 2:
                for label, delta in time_deltas:
                    time_window = qd[(stemp_time - delta < qd['time']) & (qd['time'] < stemp_time)]
                    wait_time_counts = time_window['wait_time'].value_counts()
                    if not wait_time_counts.empty:
                        wait_avr = time_window['wait_time'].sum()/len(time_window['wait_time'])
                    else:
                        wait_avr = 0
                    wait_avrs[label] = wait_avr
            else:
                time_window = qd[(stemp_time - timedelta(hours=i) < qd['time'])
                                 & (qd['time'] < stemp_time - timedelta(hours=i - 1))]
                wait_time_counts = time_window['wait_time'].value_counts()
                if not wait_time_counts.empty:
                    wait_avr = time_window['wait_time'].sum()/len(time_window['wait_time'])
                else:
                    wait_avr = 0
                wait_avrs[f'{i}h'] = wait_avr

        selected_planedata = plane_data[['flight_no', 'selected_est_out_at']]
        merged_data = pd.merge_asof(cd.sort_values('check_time'), selected_planedata.sort_values('selected_est_out_at'),
                                    by='flight_no', left_on='check_time', right_on='selected_est_out_at', direction='forward')
        merged_data['wait_time'] = merged_data['selected_est_out_at'] - merged_data['check_time']
        for i in range(1, 25):
            if i <= 2:
                for label, delta in time_deltas:
                    stemp_checkwaits[label] = merged_data[
                        (stemp_time - delta < merged_data['check_time']) & (merged_data['check_time'] < stemp_time)]
                    stemp_checkwaits[label] = stemp_checkwaits[label].fillna(stemp_checkwaits[label].mean(numeric_only=True))
            else:
                stemp_checkwaits[f'{i}h'] = merged_data[(stemp_time - timedelta(hours=i) < merged_data['check_time']) &
                                                           (merged_data['check_time'] < stemp_time - timedelta(hours=i-1))]
                stemp_checkwaits[f'{i}h'] = stemp_checkwaits[f'{i}h'].fillna(stemp_checkwaits[f'{i}h'].mean(numeric_only=True))

        time_windows = ['5', '10', '15', '20', '25', '30', '35', '40', '60', '90', '120',
                        '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', '11h', '12h', '13h', '14h',
                        '15h', '16h', '17h', '18h', '19h', '20h', '21h', '22h', '23h', '24h']
        plane_window = ['30', '60', '90', '120', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        for times in plane_window:
            planes[f'plane_in{times}'] = stemp_planes[f'stemp_plane{times}']['di'].value_counts().get('domestic', 0)
            planes[f'plane_mix{times}'] = stemp_planes[f'stemp_plane{times}']['di'].value_counts().get('mix', 0)
            planes[f'plane_t2_{times}'] = stemp_planes[f'stemp_plane{times}']['site'].value_counts().get('T2', 0)
            planes[f'plane_t3_{times}'] = stemp_planes[f'stemp_plane{times}']['site'].value_counts().get('T3', 0)
            planes[f'plane_num_{times}'] = stemp_planes[f'stemp_plane{times}']['ordered_num'].sum()
        for window in time_windows:
            # Queue
            source[f'queue_countpassed{window}'] = len(stemp_queuedatapassed[window]['queue_id'].value_counts())# 队列总数
            source[f'queue_maxpassed{window}'] = stemp_queuedatapassed[window]['queue_id'].value_counts().max()# 队列最多人数
            source[f'queue_minpassed{window}'] = stemp_queuedatapassed[window]['queue_id'].value_counts().min()#队列最少人数
            source[f'count_passed{window}'] = stemp_queuedatapassed[window]['person_counts'].sum()# 通过安检人数 数值较大 3h-21h 查queuedatapassed
            source[f'wait_max{window}'] = stemp_queuedatapassed[window]['wait_time'].value_counts().index.max()# 安检等待最长时间
            source[f'wait_min{window}'] = stemp_queuedatapassed[window]['wait_time'].value_counts().index.min()# 安检等待最短时间
            source[f'wait_avr{window}'] = wait_avrs[window]# 安检平均等待时间
            # Check
            source[f'check_countpassed{window}'] = len(stemp_checkdatas[window])
            source[f'check_T2passed{window}'] = len(stemp_checkdatas[window][stemp_checkdatas[window]['security_check_point'].str.contains('T2', na=False)])
            source[f'check_T3passed{window}'] = len(stemp_checkdatas[window]) - len(stemp_checkdatas[window][stemp_checkdatas[window]['security_check_point'].str.contains('T2', na=False)])
            source[f'check_manpassed{window}'] = stemp_checkdatas[window]['gender'].value_counts().get(1, 0)
            source[f'check_womanpassed{window}'] = stemp_checkdatas[window]['gender'].value_counts().get(2, 0)
            source[f'check_childpassed{window}'] = stemp_checkdatas[window]['gender'].value_counts().get(3, 0)
            source[f'check_waitmax{window}'] = stemp_checkwaits[window]['wait_time'].max().total_seconds()
            source[f'check_waitmin{window}'] = stemp_checkwaits[window]['wait_time'].min().total_seconds()
            source[f'check_waitavr{window}'] = stemp_checkwaits[window]['wait_time'].mean().total_seconds()
            # Traff
            source[f'traff_t2enterpassed{window}'] = stemp_trafft2datapassed[window]['enter'].sum()
            source[f'traff_t2exitpassed{window}'] = stemp_trafft2datapassed[window]['exit'].sum()
            source[f'traff_t3enterpassed{window}'] = stemp_trafft3datapassed[window]['enter'].sum()
            source[f'traff_t3exitpassed{window}'] = stemp_trafft3datapassed[window]['exit'].sum()

        source_customer.loc[len(source_customer)] = {'start_time': stemp_time,  ##时间
                                                    **source,
                                                  ## Queue
                                                  'wait_passed5': (stemp_waitpassed5['person_counts'].value_counts().index *
                                                                   stemp_waitpassed5['person_counts'].value_counts()).sum(),
                                                  'wait_passed10': (stemp_waitpassed10['person_counts'].value_counts().index *
                                                                    stemp_waitpassed10['person_counts'].value_counts()).sum(),
                                                  'wait_passed15': (stemp_waitpassed15['person_counts'].value_counts().index *
                                                                    stemp_waitpassed15['person_counts'].value_counts()).sum(),
                                                  'wait_passed20': (stemp_waitpassed20['person_counts'].value_counts().index *
                                                                    stemp_waitpassed20['person_counts'].value_counts()).sum(),
                                                  'wait_passed25': (stemp_waitpassed25['person_counts'].value_counts().index *
                                                                    stemp_waitpassed25['person_counts'].value_counts()).sum(),
                                                  'wait_passed30': (stemp_waitpassed30['person_counts'].value_counts().index *
                                                                    stemp_waitpassed30['person_counts'].value_counts()).sum(),
                                                  'wait_passed35': (stemp_waitpassed35['person_counts'].value_counts().index *
                                                                    stemp_waitpassed35['person_counts'].value_counts()).sum(),
                                                  'wait_passed40': (stemp_waitpassed40['person_counts'].value_counts().index *
                                                                    stemp_waitpassed40['person_counts'].value_counts()).sum(),
                                                  ## Plane
                                                  **planes,
                                                  ## Predict
                                                  'predict_T2_0.5':(stemp_predict30['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_1':(stemp_predict60['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_1.5':(stemp_predict90['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_2':(stemp_predict120['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_3': (stemp_predict3['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_4': (stemp_predict4['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_5': (stemp_predict5['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_6': (stemp_predict6['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_7': (stemp_predict7['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_8': (stemp_predict8['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_9': (stemp_predict9['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_10': (stemp_predict10['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_11': (stemp_predict11['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_12': (stemp_predict12['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_13': (stemp_predict13['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_14': (stemp_predict14['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_15': (stemp_predict15['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_16': (stemp_predict16['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_17': (stemp_predict17['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_18': (stemp_predict18['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_19': (stemp_predict19['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_20': (stemp_predict20['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_21': (stemp_predict21['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_22': (stemp_predict22['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_23': (stemp_predict23['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T2_24': (stemp_predict24['security_check_point'].str.split('-').str[0] == 'T2').sum(),

                                                  'predict_T3_0.5': len(stemp_predict30) - (stemp_predict30['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_1': len(stemp_predict60) - (stemp_predict60['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_1.5': len(stemp_predict90) - (stemp_predict90['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_2': len(stemp_predict120) - (stemp_predict120['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_3': len(stemp_predict3) - (stemp_predict3['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_4': len(stemp_predict4) - (stemp_predict4['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_5': len(stemp_predict5) - (stemp_predict5['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_6': len(stemp_predict6) - (stemp_predict6['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_7': len(stemp_predict7) - (stemp_predict7['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_8': len(stemp_predict8) - (stemp_predict8['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_9': len(stemp_predict9) - (stemp_predict9['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_10': len(stemp_predict10) - (stemp_predict10['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_11': len(stemp_predict11) - (stemp_predict11['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_12': len(stemp_predict12) - (stemp_predict12['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_13': len(stemp_predict13) - (stemp_predict13['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_14': len(stemp_predict14) - (stemp_predict14['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_15': len(stemp_predict15) - (stemp_predict15['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_16': len(stemp_predict16) - (stemp_predict16['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_17': len(stemp_predict17) - (stemp_predict17['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_18': len(stemp_predict18) - (stemp_predict18['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_19': len(stemp_predict19) - (stemp_predict19['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_20': len(stemp_predict20) - (stemp_predict20['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_21': len(stemp_predict21) - (stemp_predict21['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_22': len(stemp_predict22) - (stemp_predict22['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_23': len(stemp_predict23) - (stemp_predict23['security_check_point'].str.split('-').str[0] == 'T2').sum(),
                                                  'predict_T3_24': len(stemp_predict24) - (stemp_predict24['security_check_point'].str.split('-').str[0] == 'T2').sum()
                                                     }
        print(stemp_time, time.time() - stemp_count)
        if stemp_time.strftime('%Y-%m-%d %H:%M') == '2024-07-23 07:39':
            pdb.set_trace()
        stemp_time = stemp_time + timedelta(minutes=1)
    save_path = OUTPUT_DIR / f"{stemp_date.strftime('%Y%m%d')}.csv"
    source_customer.to_csv(save_path, index=False)
    print("Saved processed data to:", save_path)
    save_time = time.time()
    print(f"{stemp_date.strftime('%Y%m%d')} Saved,Runtime:", save_time-count_time)
    stemp_date = stemp_date + timedelta(days=1)