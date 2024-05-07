import numpy as np
import torch



def split_sequence_long(
    sequence: np.ndarray, ratio: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a sequence into 2 (3) parts, as is required by our transformer
    model.

    Assume our sequence length is L, we then split this into src of length N
    and tgt_y of length M, with N + M = L.
    src, the first part of the input sequence, is the input to the encoder, and we
    expect the decoder to predict tgt_y, the second part of the input sequence.
    In addition we generate tgt, which is tgt_y but "shifted left" by one - i.e. it
    starts with the last token of src, and ends with the second-last token in tgt_y.
    This sequence will be the input to the decoder.


    Args:
        sequence: batched input sequences to split [bs, seq_len, num_features]
        ratio: split ratio, N = ratio * L

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: src, tgt, tgt_y
    """
    src_end = int(sequence.shape[1] * ratio)
    # [bs, src_seq_len, num_features]
    src = sequence[:, :src_end]
    # [bs, tgt_seq_len, num_features]
    tgt = sequence[:, src_end - 1 : -1]
    # [bs, tgt_seq_len, num_features]
    tgt_y = sequence[:, src_end:]

    return src, tgt, tgt_y


def windowing_array(data, chunk_size):
    # Initialize an empty list to store the grouped time series data
    grouped_data = []
    for start in range(0, len(data), chunk_size):
        end = start + chunk_size
        # Extract the chunk of data for this group
        chunk = data[start:end]
        # Convert the chunk to a list of lists (each sublist contains heading, posx, posy, posz, etc...)
        hxyz = chunk.values.tolist()
        # Append this chunk to the grouped_data list
        grouped_data.append(hxyz)
    grouped_data_np = np.array(grouped_data, dtype=float)
    return grouped_data_np

