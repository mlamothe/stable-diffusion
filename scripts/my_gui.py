import os
import pickle
import random
import time
import threading
from tqdm import trange
from my_gui.app_options import AppOptions
from my_gui.txt2imglib import load_model, generate
from my_gui.gui import Gui


def prep_history_file():
    path_dir = os.path.expandvars(r"%OneDrive%\Documents\Stable-Diffusion")
    os.makedirs(path_dir, exist_ok=True)
    history_path_file = os.path.join(path_dir, "history.txt")
    return history_path_file


def prep_options_file():
    path_dir = os.path.expandvars(r"%APPDATA%\Stable-Diffusion")
    os.makedirs(path_dir, exist_ok=True)
    options_path_file = os.path.join(path_dir, "options.pickle")
    return options_path_file


def save_opts(options_path_file, opt):
    with open(options_path_file, "wb") as f_options:
        pickle.dump(opt, f_options)


def get_opts(options_path_file):
    if not os.path.exists(options_path_file):
        opt = AppOptions()
        save_opts(options_path_file, opt)
    else:
        with open(options_path_file, "rb") as f_options:
            opt = pickle.load(f_options)

    return opt


def append_to_history_file(history_path_file, image_file_name, opt):
    with open(history_path_file, "at", encoding='UTF-8') as f_history:
        print(
            f"{image_file_name}; Seed={opt.seed}; Steps={opt.ddim_steps}; Scale={opt.scale}; WxH={opt.W}x{opt.H}; Prompt={opt.prompt}; C={opt.C}; f={opt.f}; ddim_eta={opt.ddim_eta}; precision={opt.precision}; plms={opt.plms}; fixed_code={opt.fixed_code}",
            file=f_history,
        )


class App:
    gui = Gui()
    model: any
    history_path_file: str
    options_path_file: str
    opt: AppOptions
    dimensions: any

    def generate(self):
        self.gui.generating_started()

        self.opt = self.gui.get_opts(self.opt)
        save_opts(self.options_path_file, self.opt)

        self.generate_internal()
        self.gui.generating_completed()

    def generate_internal(self):

        artist = ''
        orig_prompt = self.opt.prompt

        if self.opt.use_artist_names == 1:
            with open(self.opt.artist_names_path, "r", encoding='UTF-8') as artist_file:
                artist_names = artist_file.readlines()
                artist_names = [line.rstrip() for line in artist_names]

        frame = 1
        orig_scale = self.opt.scale
        for _ in trange(self.opt.n_iter, desc="Sampling"):

            if self.opt.super_randomize:
                self.opt.seed = random.randrange(start=1, stop=31415928)
                #self.opt.ddim_steps = random.randrange(start=8, stop=76)
                self.opt.scale = random.randrange(start=5, stop=31)
                rnd_dimension = random.randrange(start=0, stop=len(self.dimensions))
                self.opt.W, self.opt.H = map(
                    int, self.dimensions[rnd_dimension].split("x")
                )

            if self.opt.use_artist_names:
                #artist = artist_names[frame-1]
                random.seed(time.perf_counter())
                int1 = random.randrange(start=0, stop=len(artist_names))
                random.seed(time.perf_counter())
                int2 = random.randrange(start=0, stop=len(artist_names))
                artist1 = artist_names[int1]
                artist2 = artist_names[int2]
                artist = artist1 + ' and ' + artist2
                print(f'Prompt: {self.opt.prompt} {int1} {int2}')
                self.opt.prompt = f'{orig_prompt} by {artist}'

            image_file_name = generate(self.model, self.opt, frame, artist)
            #append_to_history_file(self.history_path_file, image_file_name, self.opt)
            frame += 1
            if self.opt.increment_scale:
                self.opt.scale += 0.5                
            else:
                self.opt.seed += 1

        self.opt.scale = orig_scale

    def load_model(self):
        self.model = load_model(self.opt)
        self.gui.model_loaded()
        self.opt.n_iter = 2

        self.history_path_file = prep_history_file()

    def start(self):
        random.seed()
        self.dimensions = self.gui.dimensions()
        self.options_path_file = prep_options_file()
        self.opt = get_opts(self.options_path_file)
        print("Initial options:", self.opt)

        threading.Thread(target=self.load_model).start()

        self.gui.create_gui(self.opt, self.generate)


def main():
    App().start()


if __name__ == "__main__":
    main()
