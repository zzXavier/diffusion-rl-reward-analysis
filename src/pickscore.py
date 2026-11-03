"""Frozen PickScore features and a pairwise linear head on attribute scores."""
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

FEATURES=("prompt_cosine","mean_clause","worst_clause","clause_dispersion")


COLOR=r"red|green|blue|white|black|yellow|orange|pink|purple|brown|grey|gray|golden|silver"
NUMBER=r"one|two|three|four|five|six|seven|eight|nine|ten|\d+"
OBJECT=r"(?:bottles?|bowls?|cars?|cats?|chairs?|cups?|dogs?|hats?|houses?|lamps?|mugs?|people|persons?|rabbits?|tables?|teapots?|trees?|vases?|birds?|bicycles?|boxes|box|boats?|flowers?|horses?|castles?|paintings?|dragons?)"
SPATIAL=r"(?:(?:to\s+the\s+)?(?:left|right)\s+of|above|below|behind|under|beside|next\s+to)"


def attribute_queries(prompt):
    """Return independently scored English object, count, color and relation phrases."""
    text=prompt.casefold()
    groups=[]
    groups.extend(("object",m.group()) for m in re.finditer(r"\b"+OBJECT+r"\b",text))
    for m in re.finditer(r"\b("+NUMBER+r")\s+(?:(?:"+COLOR+r")\s+)?("+OBJECT+r")\b",text):
        groups.append(("count",f"{m[1]} {m[2]}"))
    for m in re.finditer(r"\b("+COLOR+r")\s+("+OBJECT+r")\b",text):
        groups.append(("color",f"{m[1]} {m[2]}"))
    relation=r"\b("+OBJECT+r")\s+("+SPATIAL+r")\s+(?:(?:an?|the|"+NUMBER+"|"+COLOR+r")\s+){0,3}("+OBJECT+r")\b"
    for m in re.finditer(relation,text):
        groups.append(("position",f"{m[1]} {m[2]} {m[3]}"))
    pieces=[p.strip(" .") for p in re.split(r"[,;]|\b(?:and|with|wearing|beside|behind)\b",prompt,flags=re.I)]
    if not groups:
        groups=[("clause",p) for p in pieces if p] or [("clause",prompt)]
    return list(dict.fromkeys(groups))[:20]


def clauses(prompt):
    return [phrase for _,phrase in attribute_queries(prompt)]


class FrozenScorer:
    def __init__(self, model_dir, device, prompts, processor_dir=None):
        import gc
        from transformers import AutoProcessor, CLIPConfig, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
        self.processor=AutoProcessor.from_pretrained(processor_dir or model_dir,local_files_only=True)
        self.device=device
        config=CLIPConfig.from_pretrained(model_dir,local_files_only=True)
        config.text_config.projection_dim=config.projection_dim
        config.vision_config.projection_dim=config.projection_dim
        # Both towers read exactly the same official checkpoint. Hold only one in RAM.
        text_model=CLIPTextModelWithProjection.from_pretrained(
            model_dir,config=config.text_config,local_files_only=True,low_cpu_mem_usage=True).eval().to(device)
        self.prompt_features={}
        with torch.inference_mode():
            for prompt in dict.fromkeys(prompts):
                texts=[prompt]+clauses(prompt)
                tokens=self.processor(text=texts,return_tensors="pt",padding=True,truncation=True,max_length=77).to(device)
                features=text_model(**tokens).text_embeds
                self.prompt_features[prompt]=torch.nn.functional.normalize(features.float(),dim=-1)
        del text_model
        gc.collect()
        if device=="cuda":
            torch.cuda.empty_cache()
        self.model=CLIPVisionModelWithProjection.from_pretrained(
            model_dir,config=config.vision_config,local_files_only=True,low_cpu_mem_usage=True).eval().to(device)

    @torch.inference_mode()
    def features(self,prompt,images):
        txt=self.prompt_features[prompt]
        pixels=self.processor(images=images,return_tensors="pt").to(self.device)
        img=torch.nn.functional.normalize(self.model(**pixels).image_embeds.float(),dim=-1)
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
