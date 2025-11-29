#alex' yndlingsfunktioner

import pandas as pd
import numpy as np
from Bio import SeqIO
import os
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import torch
import os
import re
import primer3
from math import ceil
import matplotlib.pyplot as plt

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

#stuff that was previously cluttering the notebook
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from math import ceil

# neccessary libraries for training
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def Initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv1d,nn.ConvTranspose1d,nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data,0,0.02)

class SelfAttentionLayer(nn.Module):
  def __init__(self,feature_in,feature_out):
    super(SelfAttentionLayer,self).__init__()
    self.Q = nn.Conv1d(feature_in, feature_out, kernel_size = 1, stride = 1, padding = 'same')
    self.K = nn.Conv1d(feature_in, feature_out, kernel_size = 1, stride = 1, padding = 'same')
    self.V = nn.Conv1d(feature_in, feature_out, kernel_size = 1, stride = 1, padding = 'same')
    self.softmax  = nn.Softmax(dim=1)
  def forward(self,x):
    Q = self.Q(x)
    K = self.K(x)
    V = self.V(x)
    # avoid creating a new CPU tensor with torch.Tensor(...)
    d = K.shape[0]
    QK_d = (Q @ K.permute(0,2,1)) / (d ** 0.5)
    prob = self.softmax(QK_d)
    attention = prob @ V
    return attention

class Generator(nn.Module):
    def __init__(self, noise_dim, gen_features, SelfAttentionLayer, out_channels=4, target_len=None):
        super(Generator, self).__init__()
        self.target_len = target_len
        self.gen = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose1d(noise_dim, gen_features, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose1d(gen_features, gen_features, kernel_size=4, stride=1, padding=1)
            ),
            nn.LeakyReLU(0.2),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose1d(gen_features, gen_features*2, kernel_size=4, stride=1, padding=1)
            ),
            nn.LeakyReLU(0.2),
            SelfAttentionLayer(gen_features*2, gen_features*2),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose1d(gen_features*2, gen_features*2, kernel_size=4, stride=1, padding=1)
            ),
            nn.LeakyReLU(0.2),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose1d(gen_features*2, gen_features*2, kernel_size=4, stride=1, padding=1)
            ),
            nn.LeakyReLU(0.2),
            SelfAttentionLayer(gen_features*2, gen_features*2),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose1d(gen_features*2, gen_features*3, kernel_size=4, stride=1, padding=1)
            ),
            nn.LeakyReLU(0.2),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv1d(gen_features*3, gen_features*3, kernel_size=4, stride=1, padding=0)
            ),
            nn.LeakyReLU(0.2),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv1d(gen_features*3, gen_features*3, kernel_size=4, stride=1, padding=1)
            ),
            nn.ReLU()
        )
        # mapping
        self.to_bases = nn.Conv1d(gen_features*3, out_channels, kernel_size=1)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device).type(torch.float32)
        out = self.gen(x)
        out = self.to_bases(out)  # now shape (B, 4, L_gen)
        if self.target_len is not None and out.size(2) != self.target_len:
            out = torch.nn.functional.interpolate(out, size=self.target_len, mode='linear', align_corners=False)
        return out
    
#critic
class Critic(nn.Module):
    def __init__(self, in_channels, disc_features, SelfAttentionLayer):
        super(Critic, self).__init__()
        # clear channel progression: in_channels -> f -> 2f -> 4f -> 1
        self.disc = nn.Sequential(
            nn.Conv1d(in_channels, disc_features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(disc_features, disc_features*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            SelfAttentionLayer(disc_features*2, disc_features*2),

            nn.Conv1d(disc_features*2, disc_features*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(disc_features*4, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device).float()
        out = self.disc(x)                  # shape (B, 1, L')
        # reduce spatial dimension to a single score per sample
        score = out.view(out.size(0), -1).mean(dim=1, keepdim=True)  # (B,1)
        return score

# gradient penalty
def GradientPenalty(critic,real,fake,device):
    Batch_size, seq_len,seq_dim = real.shape
    alpha = torch.rand(Batch_size,1,1).repeat(1,seq_len,seq_dim).to(device)
    interpolated_sequence = real * alpha + fake * (1 - alpha)
    interpolated_sequence = interpolated_sequence.to(device)

   
    # Calculate critic scores
    mixed_scores = critic(interpolated_sequence)

    # Take the gradient of the scires with respect to the sequences
    gradient = torch.autograd.grad(
        inputs = interpolated_sequence,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        retain_graph= True,
        create_graph= True,
        )[0]
    gradient = gradient.view(gradient.shape[0],-1)

    gradient_norm = gradient.norm(2,dim = 1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty



def generate_from_checkpoint(checkpoint_path="models/wgan_gp_10_epochs.pth",
                             num_sequences=10,
                             noise_dim=100,
                             gen_features=64,
                             batch_size=32,
                             target_len=None,
                             device=None,
                             seed=None,
                             temperature=1.0,
                             decode="sample"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "gen_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'gen_state_dict'")

    # instantiate generator with same architecture used in notebook
    gen = Generator(noise_dim, gen_features, SelfAttentionLayer, out_channels=4, target_len=target_len)
    try:
        gen.load_state_dict(ckpt["gen_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load generator state_dict — likely noise_dim/gen_features mismatch. "
            "Ensure noise_dim and gen_features match training. Original error: " + str(e)
        )

    gen = gen.to(device)
    gen.eval()

    converter = ["A", "C", "G", "T"]
    sequences = []
    steps = ceil(num_sequences / batch_size)

    with torch.no_grad():
        for _ in range(steps):
            cur_batch = min(batch_size, num_sequences - len(sequences))
            noise = torch.randn(cur_batch, noise_dim, 1, device=device)
            logits = gen(noise)  # shape (B, 4, L)

            # compute probabilities (temperature applied)
            probs = torch.softmax(logits / float(max(1e-8, temperature)), dim=1)  # (B,4,L)

            if decode == "argmax": #most likely base per pos
                indices = probs.argmax(dim=1)
            else:
                probs_perm = probs.permute(0, 2, 1).contiguous()  #better for different sequences
                probs2d = probs_perm.view(-1, probs_perm.size(-1)).clamp(min=1e-9)
                sampled = torch.multinomial(probs2d, num_samples=1).squeeze(1) 
                indices = sampled.view(cur_batch, probs_perm.size(1)) 
            for row in indices:
                seq = "".join(converter[int(i)] for i in row.cpu().tolist())
                sequences.append(seq)
                if len(sequences) >= num_sequences:
                    break
    return sequences

def ensure_length_match(fake, real):
    if fake.size(2) != real.size(2):
        fake = torch.nn.functional.interpolate(fake, size=real.size(2), mode='linear', align_corners=False)
    return fake

class WGANGPLossD:
    @staticmethod
    def Wasserstein(critic_real, critic_fake):
        # critic_real, critic_fake are 1D tensors (batch of scores)
        # WGAN critic loss = E[fake] - E[real]
        return critic_fake.mean() - critic_real.mean()

class WGANGPLossG:
    @staticmethod
    def Wasserstein(critic_fake):
        # generator loss = -E[critic(fake)]
        return -critic_fake.mean()
from math import ceil

def generate_from_checkpoint(checkpoint_path="models/wgan_gp_10_epochs.pth", num_sequences=10, noise_dim=100, gen_features=64, batch_size=32, target_len=None, device=None, seed=None, temperature=1.0, decode="sample"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #ensure it can run on cpu if no gpu though it will be extremely slow
    else:
        device = torch.device(device)

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "gen_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'gen_state_dict'")
    # instantiate gen from notebok
    gen = Generator(noise_dim, gen_features, SelfAttentionLayer, out_channels=4, target_len=target_len)
    try:
        gen.load_state_dict(ckpt["gen_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load generator state_dict — likely noise_dim/gen_features mismatch. "
            "Ensure noise_dim and gen_features match training. Original error: " + str(e)
        )
    gen = gen.to(device)
    gen.eval()
    converter = ["A", "C", "G", "T"]
    sequences = []
    steps = ceil(num_sequences / batch_size)
    with torch.no_grad():
        for _ in range(steps):
            cur_batch = min(batch_size, num_sequences - len(sequences))
            noise = torch.randn(cur_batch, noise_dim, 1, device=device)
            logits = gen(noise)  # shape (B, 4, L)
            # compute probabilities (temperature applied)
            probs = torch.softmax(logits / float(max(1e-8, temperature)), dim=1)  # (B,4,L)
            if decode == "argmax": #most likely base per pos
                indices = probs.argmax(dim=1)
            else:
                probs_perm = probs.permute(0, 2, 1).contiguous()  #better for different sequences
                probs2d = probs_perm.view(-1, probs_perm.size(-1)).clamp(min=1e-9)
                sampled = torch.multinomial(probs2d, num_samples=1).squeeze(1) 
                indices = sampled.view(cur_batch, probs_perm.size(1)) 
            for row in indices:
                seq = "".join(converter[int(i)] for i in row.cpu().tolist())
                sequences.append(seq)
                if len(sequences) >= num_sequences:
                    break
    return sequences

def parse_results(path:str):

    text = open(path, "r", encoding="utf-8").read()

    # capture each
    block_re = re.compile(r"(synthetic_temp[_=](\d+\.\d+),.*?)(?=synthetic_temp[_=]|\Z)", re.S | re.I)
    blocks = list(block_re.finditer(text))
    rows = []
    for m in blocks:
        block_text = m.group(1)
        temp = m.group(2)
        if re.search(r"No promoter predicted", block_text, re.I):
            score = float("nan")
            positions = []
        else:
            matches = re.findall(r"^\s*(\d+)\s+([0-9]+\.[0-9]+)\s+(.+)$", block_text, re.M)
            positions = [int(mm[0]) for mm in matches]
            scores = [float(mm[1]) for mm in matches]
            score = max(scores) if scores else float("nan")
        rows.append({
            "name": f"synthetic_temp_{temp}",
            "score": score,
            "positions": positions,
            "raw": block_text.strip()
        })
    df = pd.DataFrame(rows, columns=["name", "score", "positions", "raw"]) #parse to frame
    return df

def filter_fasta_by_names(fasta_in: str, names_df, out_fasta: str, name_col: str = "name", match_on_description: bool = False):
    nums = []
    for n in names_df[name_col].astype(str):
        m = re.search(r"(\d+\.\d+)", n)
        nums.append(m.group(1) if m else n.strip())

    nums = set(nums)
    selected = []
    found = set()

    for rec in SeqIO.parse(fasta_in, "fasta"):
        rid = rec.id or ""
        rdesc = rec.description or ""
        for num in nums:
            if num in rid or (match_on_description and num in rdesc):
                selected.append(rec)
                found.add(num)
                break

    SeqIO.write(selected, out_fasta, "fasta")
    missing = sorted(list(nums - found))
    return {"written": len(selected), "found": found, "missing": missing}

def design_primers(pth):
    for record in SeqIO.parse(pth, "fasta"):
        dna_seq = str(record.seq)
        primers = primer3.bindings.design_primers({'SEQUENCE_TEMPLATE': dna_seq,}, {'PRIMER_OPT_SIZE': 20,'PRIMER_MIN_SIZE': 18, 'PRIMER_MAX_SIZE': 25, 'PRIMER_OPT_TM': 60.0, 'PRIMER_MIN_TM': 57.0, 'PRIMER_MAX_TM': 63.0, 'PRIMER_PAIR_MAX_DIFF_TM': 3.0, 'PRIMER_PRODUCT_SIZE_RANGE': [[900, 1100]],'SEQUENCE_INCLUDED_REGION': [0, len(dna_seq)],'SEQUENCE_PRIMER_PAIR_OK_REGION_LIST': [[0, 75, len(dna_seq)-75, 75]]})
        print(f"Record: {record.id}")
        print("up primer:", primers['PRIMER_LEFT_0_SEQUENCE'])
        print("down primer:", primers['PRIMER_RIGHT_0_SEQUENCE'])
        print("product size:", primers['PRIMER_PAIR_0_PRODUCT_SIZE'])
        print("primr melting temp: ", primers['PRIMER_LEFT_0_TM'], primers['PRIMER_RIGHT_0_TM'])
        print()

#This funcion is specifically for generating a latex table for our report and probably not generally useful
def design_primers_latex_table(pth):
    for record in SeqIO.parse(pth, "fasta"):
        dna_seq = str(record.seq)
        primers = primer3.bindings.design_primers({'SEQUENCE_TEMPLATE': dna_seq},{'PRIMER_OPT_SIZE': 20,'PRIMER_MIN_SIZE': 18,'PRIMER_MAX_SIZE': 25,'PRIMER_OPT_TM': 60.0,'PRIMER_MIN_TM': 57.0,'PRIMER_MAX_TM': 63.0,'PRIMER_PAIR_MAX_DIFF_TM': 3.0,'PRIMER_PRODUCT_SIZE_RANGE': [[900, 1100]],'SEQUENCE_INCLUDED_REGION': [0, len(dna_seq)],'SEQUENCE_PRIMER_PAIR_OK_REGION_LIST': [[0, 75, len(dna_seq)-75, 75]]})
        up_seq = primers['PRIMER_LEFT_0_SEQUENCE']
        down_seq = primers['PRIMER_RIGHT_0_SEQUENCE']
        up_tm = round(primers['PRIMER_LEFT_0_TM'])
        down_tm = round(primers['PRIMER_RIGHT_0_TM'])
        print(f"\\textit{{{record.id}}} & Up & {up_tm} & 5'-{up_seq}-3' \\\\")
        print(f"& Down & {down_tm} & 5'-{down_seq}-3' \\\\")



def create_end_primers(fasta_path, out_fasta=None, primer_len=30, first_offset=20, step=30, reverse_complement=False):
    primers_by_record = {}

    for idx, rec in enumerate(SeqIO.parse(fasta_path, "fasta")):
        seq_len = len(rec.seq)
        offset = first_offset + idx * step
        start = seq_len - offset - primer_len
        if start < 0:
            start = 0 # clamp
        end = start + primer_len
        primer_seq = rec.seq[start:end]
        if reverse_complement:
            primer_seq = primer_seq.reverse_complement()
        primers_by_record[rec.id] = [(offset, int(start), int(end), str(primer_seq))]

    if out_fasta:
        out_records = []
        for rec_id, plist in primers_by_record.items():
            for offset, start, end, pseq in plist:
                rid = f"{rec_id}_off{offset}"
                out_records.append(SeqRecord(Seq(pseq), id=rid, description=f"start={start};end={end};offset={offset}"))
        out_dir = os.path.dirname(out_fasta)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        SeqIO.write(out_records, out_fasta, "fasta")
    return primers_by_record

def tensor_tile_for_ppt(
    tensor,
    seq_selector=slice(0, 4),
    pos_slice=(0, 80),
    channel_order="ACGT",
    cmap="viridis",
    out_png=None,
    layout="horizontal",  # "horizontal" (default) or "stack"
    return_decoded=False,
):
    """
    Create a tiled image of parts of a one-hot tensor for PowerPoint.

    - layout: "horizontal" (concatenate sequences side-by-side) or "stack" (stack sequences vertically)
    - seq_selector: int, slice, list/ndarray of indices, or callable -> selected indices
    - pos_slice: (start, end) positions to display
    - return_decoded: also return decoded sequence strings for displayed slices
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    N, C, L = arr.shape
    assert len(channel_order) == C, "channel_order length must equal channel count"

    # resolve seq_selector -> list of indices
    if isinstance(seq_selector, int):
        indices = list(range(min(seq_selector, N)))
    elif isinstance(seq_selector, slice):
        indices = list(range(N))[seq_selector]
    elif isinstance(seq_selector, (list, tuple, np.ndarray)):
        indices = list(seq_selector)
    elif callable(seq_selector):
        indices = list(seq_selector(arr))
    else:
        raise TypeError("seq_selector must be int, slice, list/ndarray or callable")

    indices = [i for i in indices if isinstance(i, int) and 0 <= i < N]
    if len(indices) == 0:
        raise ValueError("No valid sequence indices selected")

    start = max(0, int(pos_slice[0]))
    end = min(L, int(pos_slice[1]))
    Ls = max(1, end - start)
    m = len(indices)

    # build tiles
    tiles = [arr[i, :, start:end] for i in indices]  # each shape (C, Ls)

    if layout == "horizontal":
        big = np.concatenate(tiles, axis=1)  # shape (C, m*Ls)
        fig, ax = plt.subplots(figsize=(max(6, m * 2), max(2, C * 0.5)))
        im = ax.imshow(big, aspect="auto", cmap=cmap)
        ax.set_yticks(np.arange(C))
        ax.set_yticklabels(list(channel_order))
        ax.set_xlabel("position (concatenated sequences)")
        ax.set_title(f"One-hot channels for sequences {indices} (positions {start}:{end})")
        # vertical dividers & labels
        for i in range(1, m):
            x = i * Ls - 0.5
            ax.axvline(x, color="white", linewidth=2)
        #for i, seq_idx in enumerate(indices):
        #    cx = i * Ls + Ls / 2
        #    ax.text(cx, -0.6, f"seq {seq_idx}", ha="center", va="bottom", fontsize=10, transform=ax.get_xaxis_transform())

    elif layout == "stack":
        # stack vertically: big shape (m*C, Ls)
        big = np.vstack(tiles)
        fig, ax = plt.subplots(figsize=(max(6, Ls / 20), max(2, m * C * 0.25)))
        im = ax.imshow(big, aspect="auto", cmap=cmap)
        # y ticks: repeat channel labels for each sequence
        y_positions = np.arange(m * C)
        y_labels = []
        for seq_idx in indices:
            for ch in channel_order:
                y_labels.append(ch)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("position")
        ax.set_title(f"Stacked one-hot channels for sequences {indices} (positions {start}:{end})")
        # horizontal separators between sequences
        for i in range(1, m):
            y = i * C - 0.5
            ax.axhline(y, color="white", linewidth=2)
        # annotate sequence index left of each block
        #for i, seq_idx in enumerate(indices):
        #    y_mid = i * C + (C - 1) / 2.0
        #    ax.text(-0.02 * Ls, y_mid, f"seq {seq_idx}", ha="right", va="center", fontsize=10, transform=ax.get_yaxis_transform())

    else:
        raise ValueError("layout must be 'horizontal' or 'stack'")

    ax.set_xlim(-0.5, Ls - 0.5)
    #plt.tight_layout(rect=[0, 0.03, 1, 1])
    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")

    return fig

__all__ = [
    "extract_upstream_sequences",
    "one_hot_encode_sequence_to_tensor",
    "one_hot_encode_df_to_tensor",
    "Initialize_weights",
    "SelfAttentionLayer",
    "Generator",
    "Critic",
    "GradientPenalty",
    "generate_from_checkpoint",
    "ensure_length_match",
    "WGANGPLossD",
    "WGANGPLossG",
    "parse_results",
    "filter_fasta_by_names",
    "design_primers",
    "create_end_primers",
    "tensor_tile_for_ppt",
]