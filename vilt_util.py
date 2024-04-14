import re

import torch

from vilt.transforms import pixelbert_transform


def predict_text_and_embedding(image, mp_text, tokenizer, vilt_model, text_encoder_model):
    device = "cuda:0"
    vilt_model.to(device)
    batch = {"text": [mp_text], "image": [None]}
    tl = len(re.findall("\[MASK\]", mp_text))
    inferred_token = [mp_text]

    img = pixelbert_transform(size=384)(image)
    img = img.unsqueeze(0).to(device)
    batch["image"][0] = img

    with torch.no_grad():
        for i in range(tl):
            batch["text"] = inferred_token
            encoded = tokenizer(inferred_token)
            batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
            encoded = encoded["input_ids"][0][1:-1]
            infer = vilt_model(batch)
            mlm_logits = vilt_model.mlm_score(infer["text_feats"])[0, 1:-1]
            mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
            mlm_values[torch.tensor(encoded) != 103] = 0
            select = mlm_values.argmax().item()
            encoded[select] = mlm_ids[select].item()
            inferred_token = [tokenizer.decode(encoded)]

    text_encoder_model.to(device)

    encoded_text = tokenizer(inferred_token[0], return_tensors="pt")
    encoded_text = {key: tensor.to(device) for key, tensor in encoded_text.items()}
    text_embedding = text_encoder_model(**encoded_text).last_hidden_state.mean(dim=1)

    return inferred_token[0], text_embedding
