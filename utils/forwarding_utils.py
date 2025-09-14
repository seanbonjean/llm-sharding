import torch


def build_position_ids(past_key_value, seq_len, device, batch_size: int = 1):
    """
    past_key_value:
      - HF Cache:  可用 past_key_value.get_seq_length()
      - (k, v)元组: 用 k.shape[-2] 作为已缓存长度 (k: [B, n_kv, past_len, Hd])
      - None: past_len = 0
    返回:
      position_ids: [B, S] LongTensor
    """
    if past_key_value is None:
        past_len = 0
    elif hasattr(past_key_value, "get_seq_length"):
        past_len = int(past_key_value.get_seq_length())
    elif isinstance(past_key_value, tuple) and len(past_key_value) == 2:
        k = past_key_value[0]
        past_len = int(k.shape[-2])
    else:
        # 如果你自定义了 cache 结构，这里替换成正确的取法
        raise ValueError("[ERROR] Unsupported past_key_value structure")

    pos = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long)  # [S]
    position_ids = pos.unsqueeze(0).expand(batch_size, -1).contiguous()  # [B, S]
    return position_ids
