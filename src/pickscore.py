"""Frozen PickScore features and a pairwise linear head on attribute scores."""
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

FEATURES=("prompt_cosine","mean_clause","worst_clause","clause_dispersion")


def clauses(prompt):
    pieces=[p.strip(" .") for p in re.split(r"[,;]|\b(?:and|with|wearing|beside|behind)\b",prompt,flags=re.I)]
    return list(dict.fromkeys(p for p in pieces if p))[:8] or [prompt]


class FrozenScorer:
    def __init__(self, model_dir, device):
        from transformers import AutoProcessor, CLIPModel
        self.processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
        self.model=CLIPModel.from_pretrained(model_dir,local_files_only=True).eval().to(device)
        self.device=device

    @torch.inference_mode()
    def features(self,prompt,images):
        texts=[prompt]+clauses(prompt)
        t=self.processor(text=texts,return_tensors="pt",padding=True,truncation=True,max_length=77).to(self.device)
        i=self.processor(images=images,return_tensors="pt").to(self.device)
        txt=torch.nn.functional.normalize(self.model.get_text_features(**t),dim=-1)
        img=torch.nn.functional.normalize(self.model.get_image_features(**i),dim=-1)
        scores=(img@txt.T).float().cpu().numpy()
        return np.column_stack([scores[:,0],scores[:,1:].mean(1),scores[:,1:].min(1),scores[:,1:].std(1)])


def head_from(rows):
    diffs=[]
    for row in rows:
        if row["split"]=="train":
            values=np.asarray(row["features"])
            diffs.append(values[row["winner"]]-values[1-row["winner"]])
    if not diffs:
        raise ValueError("no training pairs")
    x=np.asarray(diffs)
    scale=np.maximum(x.std(axis=0),1e-5)
    inputs=np.r_[x/scale,-x/scale]
    labels=np.r_[np.ones(len(x)),np.zeros(len(x))]
    model=LogisticRegression(C=0.2,fit_intercept=False,solver="lbfgs").fit(inputs,labels)
    return dict(weights=(model.coef_[0]/scale).tolist(), features=list(FEATURES))


def prediction_sets(row,head):
    x=np.asarray(row["features"])
    return {"first":0,"whole":int(x[:,0].argmax()),"clauses":int(x[:,1].argmax()),
            "preference_head":int((x@np.asarray(head["weights"])).argmax())}
