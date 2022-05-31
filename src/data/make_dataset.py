import tensorflow as tf

def get_eeg() -> Path:
    """
    This function downloads the eeg dataset from:
    https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
    and stores it in the given data dir. 
    """
    data_dir = "../../data/raw"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
    datapath = tf.keras.utils.get_file(
        "eeg", origin=url, untar=False, cache_dir=data_dir
    )

    return data_dir