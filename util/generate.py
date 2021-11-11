import torch

def text_generate(model, tok, vocab, sent, text_size, temperature, top_p, top_k):
    input_ids = tok.encode(sent)
    gen_ids = model.generate(torch.tensor([input_ids], device='cuda'),
                           max_length=200,
                           repetition_penalty=2.0,
                           pad_token_id=tok.pad_token_id,
                           eos_token_id=tok.eos_token_id,
                           bos_token_id=tok.bos_token_id,
                           use_cache=True)

    generated = tok.decode(gen_ids[0,:].tolist())

    return generated