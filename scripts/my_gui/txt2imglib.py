import os
import sys
import time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from my_gui.app_options import AppOptions


def load_model(opt: AppOptions):
    config = OmegaConf.load(f"{opt.config}")
    model = _load_model_from_config(config, f"{opt.ckpt}")
    return model


def generate(model, opt: AppOptions, frame=1) -> str:
    print(f"opt = {opt}")

    seed_everything(opt.seed)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else sys.exit("No GPU found")
    )
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt
    assert prompt is not None
    data = [[prompt]]

    sample_path = outpath
    os.makedirs(sample_path, exist_ok=True)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn(
            [1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device
        )

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(1 * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=start_code,
                    )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                    )
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image = x_samples_ddim

                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(
                        0, 3, 1, 2
                    )

                    for x_sample in x_checked_image_torch:
                        x_sample = 255.0 * rearrange(
                            x_sample.cpu().numpy(), "c h w -> h w c"
                        )
                        img = Image.fromarray(x_sample.astype(np.uint8))

                        if opt.mov_file_names:
                            filename = f"frame_{frame:05}.png"
                        else:
                            timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
                            filename = f"{timestr}-seed-{opt.seed}-scale-{opt.scale}-steps-{opt.ddim_steps}.png"

                        img.save(os.path.join(sample_path, filename))
    return filename


def _load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model
