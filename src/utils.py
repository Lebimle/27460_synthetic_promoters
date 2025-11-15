import pandas as pd
from Bio import SeqIO
import os
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os

# function for extracting the upstream sequences
def extract_upstream_sequences(gbk_path:str, upstream_length:int=1000, feature_types:list=("CDS")):
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