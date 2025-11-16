import pandas as pd
from Bio import SeqIO
import os
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import os
import numpy as np
import torch

# function for extracting the upstream sequences
def extract_upstream_sequences(gbk_path:str, upstream_length:int=1000, feature_types:list=["CDS"]):
    sequences = []
    ids = []

    for rec in SeqIO.parse(gbk_path, "genbank"):
        for feature in rec.features:
            if feature.type in feature_types:
                start = feature.location.start
                if start >= upstream_length:
                    upstream_seq = rec.seq[start - upstream_length:start]
                else:
                    upstream_seq = rec.seq[:start]
                sequences.append(str(upstream_seq))
                ids.append(feature.qualifiers.get('locus_tag', ['unknown'])[0])
    df = pd.DataFrame({'id': ids, 'sequence': sequences})
    return df

# function for one-hot encoding sequences to tensor for ML training
def one_hot_encode_sequence_to_tensor(seqs: str, seq_len: int) -> np.ndarray:
    mapping = {'A':0,'C':1,'G':2,'T':3}
    N = len(seqs)
    arr = np.zeros((N, 4, seq_len), dtype=np.float32) #initialize array
    for i, seq in enumerate(seqs):
        for j, nucleotide in enumerate(seq):
            if nucleotide in mapping: # leave all zeros if unknown
                arr[i, mapping[nucleotide], j] = 1.0
    tensor = torch.from_numpy(arr)
    return tensor

def one_hot_encode_df_to_tensor(df, seq_len, seq_col) -> torch.Tensor:
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    seqs = df[seq_col].astype(str).tolist()
    N = len(seqs)
    arr = np.zeros((N, 4, seq_len), dtype = np.float32)

    for i, seq in enumerate(seqs):
        for j, nucleotide in enumerate(seq[:seq_len]):
            if nucleotide in mapping:
                arr[i, mapping[nucleotide], j] = 1.0

    tensor = torch.from_numpy(arr)
    return tensor

