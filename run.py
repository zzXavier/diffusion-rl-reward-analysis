"""Real DDPO rollouts and clipped policy updates using the pinned upstream log-probability scheduler."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageDraw
from diffusers import DDIMScheduler
from peft import set_peft_model_state_dict
from src.ddpo.ddim_with_logprob import ddim_step_with_logprob
from src.pickscore import FrozenScorer
from tools.pickscore_processor import fetch_processor
from src.policy import clipped_loss,color_fraction,roughness,standardized
from src.sd import add_adapter,encode,load_pipeline,save_reload,seed_all,trainable_state

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/"configs/experiment.json").read_text())


@torch.no_grad()
def trajectory(pipe,prompt,seed,steps,size):
    scheduler=pipe.scheduler
    scheduler.set_timesteps(steps,device=pipe.device)
    gen=torch.Generator(device=pipe.device).manual_seed(seed)
    shape=(1,pipe.unet.config.in_channels,size//pipe.vae_scale_factor,size//pipe.vae_scale_factor)
    x=torch.randn(shape,generator=gen,device=pipe.device,dtype=pipe.unet.dtype)
    context=encode(pipe,[prompt])
    states=[]
    for i,t in enumerate(scheduler.timesteps):
        times=t.reshape(1)
        noise=pipe.unet(x,times,context).sample
        next_x,logp=ddim_step_with_logprob(scheduler,noise.float(),times,x.float(),eta=1.,generator=gen)
        prev_for_logprob=next_x.detach()
        next_x=next_x.to(x.dtype)
        if i<len(scheduler.timesteps)-1:
            if not torch.isfinite(logp).all():
                raise ValueError("non-finite stochastic transition probability")
            states.append((times.detach(),x.detach(),prev_for_logprob,logp.detach()))
        x=next_x
    output=pipe.vae.decode(x/pipe.vae.config.scaling_factor).sample
    image=pipe.image_processor.postprocess(output,output_type="pil")[0]
    return image,context,states


def evaluate(out):
    records=json.loads((out/"predictions.json").read_text())
    prompt_to_channel=dict(CFG["test_prompts"])
    test_index={prompt:i for i,(prompt,_) in enumerate(CFG["test_prompts"])}
    for row in records:
        frame=[]
        for i in range(row["candidates"]):
            name=f"{row['method']}-{test_index[row['prompt']]}-{i}.png"
            with Image.open(out/name) as image:
                frame.append(np.array(image.convert("RGB"),dtype=float)/255.)
        with Image.open(out/row["selected_file"]) as selected:
            actual_color=color_fraction(selected,prompt_to_channel[row["prompt"]])
        diversity=float(np.mean([np.abs(frame[i]-frame[j]).mean() for i in range(len(frame)) for j in range(i)]))
        if not np.isclose(actual_color,row["independent_color_fraction"],rtol=0,atol=1e-12) or not np.isclose(diversity,row["pairwise_pixel_l1"],rtol=0,atol=1e-12):
            raise ValueError("Saved outputs differ from recorded independent diagnostics")
        row["independent_color_fraction"]=actual_color
        row["pairwise_pixel_l1"]=diversity
    summary=[]
    for name in dict.fromkeys(r["method"] for r in records):
        rows=[r for r in records if r["method"]==name]
        summary.append({"method":name,"prompts":len(rows),
                        **{k:float(np.mean([r[k] for r in rows])) for k in ["selected_reward","independent_color_fraction","pairwise_pixel_l1"]}})
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    cols=len(summary)
    sheet=Image.new("RGB",(cols*160,180),"white")
    d=ImageDraw.Draw(sheet)
    for i,r in enumerate(summary):
        d.text((i*160+2,3),r["method"],fill="black")
        name=next(v["selected_file"] for v in records if v["method"]==r["method"])
        with Image.open(out/name) as im:
            sheet.paste(im.convert("RGB").resize((152,152)),(i*160+3,25))
    sheet.save(out/"comparison.png")
    print((out/"summary.csv").read_text())


def main(args):
    smoke=args.mode=="demo"
    out=args.out or ROOT/("demo-output" if smoke else "results")
    if args.mode=="evaluate":
        return evaluate(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    n=2 if smoke else CFG["candidates"]
    steps=3 if smoke else CFG["sample_steps"]
    size=32 if smoke else CFG["resolution"]
    judge=None
    weights=None
    if not smoke:
        if args.pickscore is None:
            from tools.get_model import fetch
            args.pickscore=fetch(args.cache/"PickScore")
        with (args.pickscore/"model.safetensors").open("rb") as handle:
            if hashlib.file_digest(handle,"sha256").hexdigest()!="ef31ef6fc5ff4d9bb90dd232df4e145887ba62c5a03aa2841415f8c25f18d52e":
                raise ValueError("P02 scorer checkpoint differs from the pinned PickScore v1 model")
        if hashlib.sha256(args.head.read_bytes()).hexdigest()!="b9cd5ee6aa4ca6cb749a291ddfbf3c3da56e39559b38f54fe9d0418f2fd6a597":
            raise ValueError("P02 head differs from the frozen release head")
        raw=json.loads(args.head.read_text())
        weights=np.asarray(raw["weights"])
        if raw["features"]!=["prompt_cosine","mean_clause","worst_clause","clause_dispersion"] or not np.isfinite(weights).all():
            raise ValueError("Unexpected P02 head schema")
        judge=FrozenScorer(args.pickscore,args.reward_device,[p for p,_ in CFG["train_prompts"]+CFG["test_prompts"]],
                           fetch_processor(args.cache/"PickScore-processor"))
    def rewards(prompt,images):
        if smoke:
            # Fast test-only reward. It has no image/text preference meaning.
            return [roughness(im) for im in images]
        return (judge.features(prompt,images)@weights).tolist()
    pipe=load_pipeline(args.device,smoke,args.cache,args.base)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config,clip_sample=False)
    seed_all(CFG["seed"])
    params=add_adapter(pipe.unet,CFG["rank"])
    initial=trainable_state(pipe.unet)
    predictions=[]
    base_candidates={}
    logs=[]
    updates={}
    for method in ["base","rerank","single-ddpo","constrained-ddpo"]:
        set_peft_model_state_dict(pipe.unet,initial)
        if method.endswith("ddpo"):
            opt=torch.optim.AdamW(params,lr=CFG["learning_rate"])
            for epoch in range(2 if smoke else CFG["epochs"]):
                print(f"{method}: rollout epoch {epoch+1}",flush=True)
                prompt,_=CFG["train_prompts"][epoch%len(CFG["train_prompts"])]
                rollout=[trajectory(pipe,prompt,CFG["seed"]+epoch*10+i,steps,size) for i in range(n)]
                images=[v[0] for v in rollout]
                reward=np.asarray(rewards(prompt,images))
                if method.startswith("constrained"):
                    reward-=.05*np.asarray([roughness(im) for im in images])
                adv=standardized(reward)
                for inner_epoch in range(2):
                    opt.zero_grad(set_to_none=True)
                    for j,(_,ctx,states) in enumerate(rollout):
                        for t,x,prev,old in states:
                            noise=pipe.unet(x,t,ctx).sample
                            _,logp=ddim_step_with_logprob(pipe.scheduler,noise.float(),t,x.float(),eta=1.,prev_sample=prev.float())
                            loss=clipped_loss(logp,old,float(adv[j]),CFG["clip_range"])/(n*len(states))
                            loss.backward()
                            logs.append(dict(method=method,epoch=epoch,inner_epoch=inner_epoch,sample=j,step=int(t.item()),
                                             reward=float(reward[j]),advantage=float(adv[j]),old_logp=float(old.item()),
                                             new_logp=float(logp.detach().item()),loss=float(loss.detach())))
                    torch.nn.utils.clip_grad_norm_(params,1.)
                    opt.step()
            updates[method]=save_reload(pipe.unet,initial,out/f"{method}.safetensors")
        for k,(prompt,hue) in enumerate(CFG["test_prompts"][:1 if smoke else None]):
            # Every arm draws the same n images per evaluation prompt.
            if method=="rerank":
                # Exactly the same generated candidate bank as the frozen baseline.
                imgs,score=base_candidates[k]
            else:
                imgs=[trajectory(pipe,prompt,CFG["seed"]+999+k*10+i,steps,size)[0] for i in range(n)]
                score=rewards(prompt,imgs)
                if method=="base":
                    base_candidates[k]=(imgs,score)
            chosen=int(np.argmax(score)) if method=="rerank" else 0
            files=[]
            for j,img in enumerate(imgs):
                f=f"{method}-{k}-{j}.png"
                img.save(out/f)
                files.append(f)
            x=np.asarray([np.asarray(im,dtype=float)/255. for im in imgs])
            div=np.mean([np.abs(x[i]-x[j]).mean() for i in range(n) for j in range(i)])
            predictions.append(dict(method=method,prompt=prompt,candidates=n,selected_file=files[chosen],
                                    selected_reward=score[chosen],independent_color_fraction=color_fraction(imgs[chosen],hue),
                                    pairwise_pixel_l1=float(div),rewards=score))
    (out/"predictions.json").write_text(json.dumps(predictions,indent=2))
    (out/"policy-log.json").write_text(json.dumps(logs,indent=2))
    (out/"run.json").write_text(json.dumps(dict(profile=args.mode,updates_l2=updates,
                   reward="smoke-total-variation" if smoke else "P02-frozen",
                   reward_device="cpu" if smoke else args.reward_device,
                   reward_head_sha256=hashlib.sha256(args.head.read_bytes()).hexdigest() if not smoke else None,
                   torch=str(torch.__version__),hardware=torch.cuda.get_device_name() if pipe.device.type=="cuda" else "cpu",
                   training_trajectories_per_policy=n*(2 if smoke else CFG["epochs"])),indent=2))
    evaluate(out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["demo","reproduce","evaluate"])
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache",type=Path,default=ROOT/".cache")
    p.add_argument("--base",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--pickscore",type=Path)
    p.add_argument("--reward-device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--head",type=Path,default=ROOT/"examples/p02-head.json")
    main(p.parse_args())
