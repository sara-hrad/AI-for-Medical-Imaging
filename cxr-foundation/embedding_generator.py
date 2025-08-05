import os
import shutil
import pandas as pd
import logging
from google.cloud import storage
from utils import *

from cxr_foundation.inference import ModelVersion, generate_embeddings, InputFileType, OutputFileType


def download_images(file_name, DICOM_DIR):
    # Initialize the GCS storage client
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('cxr-foundation-demo')
    stats = storage.Blob(bucket=bucket, name='cxr14/labels.csv').exists(storage_client)
    df = pd.to_csv(file_name)
    df["remote_dicom_file"] = df["dicom_file"].apply(
    lambda x: os.path.join('cxr14/inputs', os.path.basename(x)))
    if not os.path.exists(DICOM_DIR):
        os.makedirs(DICOM_DIR)

    for _, row in df.iterrows():
        blob = bucket.blob(row["remote_dicom_file"])
        print(row)
        if blob.exists():
            print(row["dicom_file"])
            blob.download_to_filename(row["dicom_file"])
    print("Finished downloading DICOM files!")

def get_embeddings(EMBEDDINGS_DIR, filename):
    EMBEDDING_VERSION = 'cxr_foundation' #@param ['elixr', 'cxr_foundation', 'elixr_img_contrastive']
    if EMBEDDING_VERSION == 'cxr_foundation':
        MODEL_VERSION = ModelVersion.V1
    elif EMBEDDING_VERSION == 'elixr':
        MODEL_VERSION = ModelVersion.V2
    elif EMBEDDING_VERSION == 'elixr_img_contrastive':
        MODEL_VERSION = ModelVersion.V2_CONTRASTIVE
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)
    else:
        # Empty embedding dir to avoid caching when switching embedding versions
        shutil.rmtree(EMBEDDINGS_DIR)
        os.makedirs(EMBEDDINGS_DIR)

    df_labels = pd.read_csv(filename)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    generate_embeddings(input_files=df_labels["dicom_file"].values, output_dir=EMBEDDINGS_DIR,
                        input_type=InputFileType.DICOM, output_type=OutputFileType.TFRECORD,
                        model_version=MODEL_VERSION)


if __name__=="__main__":
    MAX_CASES_PER_CATEGORY = 1000  # @param {type: 'integer'}
    DICOM_DIR = 'data/inputs'  # @param {type: 'string'}
    EMBEDDINGS_DIR = 'data/outputs'  # @param {type: 'string'}

    dataset_name = DatasetType.Abnormal_1000    # Either use Abnormal_1000 or Normal_1000 for dl the data (dataset is the same)
    dataset_dir, filename, DIAGNOSIS = dataset_directory(dataset_name)
    download_images(file_name=filename, DICOM_DIR =os.path.join(dataset_dir, DICOM_DIR))
    get_embeddings(EMBEDDINGS_DIR = os.path.join(dataset_dir, DICOM_DIR), filename=filename)