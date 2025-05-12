import pandas as pd
import networkx as nx
import os
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, AutoFeatureExtractor, WavLMModel
import torch
import librosa
import hdbscan
import numpy as np
from datetime import datetime
# import plotly.express as px
# from pyannote.audio import Inference, Model
import glob
from tqdm import tqdm
# from scipy.spatial.distance import cdist
import random

#from itables import init_notebook_mode, show as Show
#init_notebook_mode(connected=False)


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(f"Using device: {device}")

METADATA_FILE = "metadata.csv"
CLUSTER_GRAPH = "graph-clusters.gpickle"
ROBOCALL_AUDIO_DIRECTORY_NORMALIZED_16k = "FTC-raw-audio-ppone-normalized/"
SAMPLING_RATE_16k = 16000


G = nx.read_gpickle(CLUSTER_GRAPH)
subg = sorted(list(nx.connected_components(G)), key=len, reverse=True)

df = pd.read_csv(METADATA_FILE)
df['fn'] = df['file_name'].apply(lambda x: os.path.basename(x))
df['clusterid'] = pd.Series([-1] * df.shape[0])

for clusterid, cluster in enumerate(subg):
    for fn in cluster:
        df.loc[df['fn'] == fn.replace("_normalized.json", ".wav"), 'clusterid'] = clusterid


audio_paths = glob.glob(ROBOCALL_AUDIO_DIRECTORY_NORMALIZED_16k + "*_normalized.wav") + glob.glob(ROBOCALL_AUDIO_DIRECTORY_NORMALIZED_16k + "*_left.wav")



audio_paths_processed = []
embeddings = []
model_name_wav2vec2 = "facebook/wav2vec2-base-960h"
feature_extractor_wav2vec2 = Wav2Vec2FeatureExtractor.from_pretrained(model_name_wav2vec2)
model_wav2vec2 = Wav2Vec2Model.from_pretrained(model_name_wav2vec2)
model_wav2vec2 = model_wav2vec2.to('cuda:0')

for audio_ in tqdm(audio_paths):

  try:
    input_audio, sample_rate = librosa.load(audio_,  sr=SAMPLING_RATE_16k)
    audio_input_vector = feature_extractor_wav2vec2(input_audio, return_tensors="pt", sampling_rate=SAMPLING_RATE_16k)
    audio_input_vector = audio_input_vector.to(device)
    with torch.no_grad():
      model_output = model_wav2vec2(audio_input_vector.input_values, output_hidden_states=True)
    embedding = model_output.last_hidden_state.mean(dim=1)
    #move output to CPU for analysis
    embeddings.append(embedding.cpu())
    audio_paths_processed.append(audio_)
  except Exception as err:
    #Some files may fail because they are empty
    #Example below
    #Failed to load data_100_robocalls/368913.wav
    #Unexpected err=LibsndfileError(1, "Error opening 'data_100_robocalls/368913.wav': "), type(err)=<class 'soundfile.LibsndfileError'>
    print("Failed to load {}".format(audio_))
    print(f"Unexpected {err=}, {type(err)=}")


audio_embedding_df = pd.DataFrame({"audio_path": audio_paths_processed,
                                   "audio_embedding_wav2vec2": embeddings})
print(audio_embedding_df.head())

audio_embedding_df['fn_no_extension'] = audio_embedding_df['audio_path'].apply(lambda x: os.path.basename(x).replace("_normalized.wav", "").replace("_left.wav", ""))
df['fn_no_extension'] = df['fn'].apply(lambda x: os.path.basename(x).rstrip(".wav"))
df = pd.merge(df, audio_embedding_df,
                how = "inner", on = "fn_no_extension")


audio_paths_processed = []
embeddings = []
model_name_wavlm = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
feature_extractor_wavlm = AutoFeatureExtractor.from_pretrained(model_name_wavlm)
model_wavlm = WavLMModel.from_pretrained(model_name_wavlm)
model_wavlm = model_wavlm.to(device)


for audio_ in tqdm(audio_paths):

  try:
    SAMPLING_RATE = 16000
    input_audio, sample_rate = librosa.load(audio_,  sr=SAMPLING_RATE_16k)
    audio_input_vector = feature_extractor_wavlm(input_audio, return_tensors="pt", sampling_rate=SAMPLING_RATE_16k)
    audio_input_vector = audio_input_vector.to(device)
    with torch.no_grad():
      model_output = model_wavlm(audio_input_vector.input_values, output_hidden_states=True)
    embedding = model_output.last_hidden_state.mean(dim=1)
    #move output to CPU for analysis
    embeddings.append(embedding.cpu())
    audio_paths_processed.append(audio_)
  except Exception as err:
    #Some files may fail because they are empty
    #Example below
    #Failed to load data_100_robocalls/368913.wav
    #Unexpected err=LibsndfileError(1, "Error opening 'data_100_robocalls/368913.wav': "), type(err)=<class 'soundfile.LibsndfileError'>
    print("Failed to load {}".format(audio_))
    print(f"Unexpected {err=}, {type(err)=}")

audio_embedding_df = pd.DataFrame({"audio_path": audio_paths_processed,
                                   "audio_embedding_wavlm": embeddings})
audio_embedding_df['fn_no_extension'] = audio_embedding_df['audio_path'].apply(lambda x: os.path.basename(x).replace("_normalized.wav", "").replace("_left.wav", ""))


df['fn_no_extension']=df['fn_no_extension'].astype(str)
audio_embedding_df['fn_no_extension']=audio_embedding_df['fn_no_extension'].astype(str)

df_m = pd.merge(df, audio_embedding_df,
                how = "inner", on = "fn_no_extension")
df_m.to_pickle("df_echoprint_and_wav2vec2_and_wavlm_embeddings_run_two.pkl")

df_m['audio_embedding_wav2vec2_reshape'] = df_m['audio_embedding_wav2vec2'].map(lambda x: np.array(x)[0])

clusterer = hdbscan.HDBSCAN(min_cluster_size = 2)
resp = clusterer.fit(df_m['audio_embedding_wav2vec2_reshape'].to_list())

df_m['clusterid_hdbscan_audio_embedding_wav2vec2_reshape'] = pd.Series(resp.labels_)

df_m['audio_embedding_wavlm_reshape'] = df_m['audio_embedding_wavlm'].map(lambda x: np.array(x)[0])

clusterer = hdbscan.HDBSCAN(min_cluster_size = 2)
resp = clusterer.fit(df_m['audio_embedding_wavlm_reshape'].to_list())

df_m['clusterid_hdbscan_audio_embedding_wavlm_reshape'] = pd.Series(resp.labels_)


def print_cluster_groups(df, cluster_id_col):
    # Group the dataframe based on the values in "ClusterID"
    grouped_df = df.groupby(cluster_id_col)

    # Sort the groups in decending order of size (largest first, smallest last)
    sorted_groups = grouped_df.size().sort_values(ascending=False).index

    # For each group, print the dataframe
    with open(cluster_id_col + "_cluster.log", "w") as f:
        for group in sorted_groups:
            if group == -1:
                continue
            f.write(f"====Cluster {group}====:\n")
            f.write("\n".join(sorted([str(x) for x in grouped_df.get_group(group)['transcript'].to_list()]
                                     , key = lambda x: len(x),
                                     reverse=True)))
            f.write("\n\n\n")

        #Write the non-custered group
        group = -1
        f.write(f"====Non-Clustered-Transcripts clusterID: {group} | Count: {grouped_df.get_group(group).shape[0]} ====:\n")
        f.write("\n".join(sorted([str(x) for x in grouped_df.get_group(group)['transcript'].to_list()]
                                    , key = lambda x: len(x),
                                    reverse=True)))
        f.write("\n\n\n")

print_cluster_groups(df_m, 'clusterid')
print_cluster_groups(df_m, 'clusterid_hdbscan_audio_embedding_wavlm_reshape')
print_cluster_groups(df_m, 'clusterid_hdbscan_audio_embedding_wav2vec2_reshape')
