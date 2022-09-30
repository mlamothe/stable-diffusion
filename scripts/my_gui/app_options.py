from dataclasses import dataclass


@dataclass
class AppOptions:
    prompt: str = "a painting of a virus monster playing guitar"
    seed: int = 31415927
    config: str = "c:/src/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    ckpt: str = "c:/src/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
    artist_names_path: str = "D:/ai_art/artist_names.txt"
    plms: bool = False
    outdir: str = "D:/ai_art/sd_local/"
    fixed_code: bool = False
    ddim_steps: int = 10
    scale: float = 7.5
    n_iter: int = 1
    H: int = 512
    W: int = 512
    C: int = 4
    f: int = 8
    precision: str = "autocast"
    ddim_eta: float = 0.0

    # Extras for the UI
    mov_file_names: int = 0
    increment_scale: int = 0
    super_randomize: int = 0
    use_artist_names: int = 0
