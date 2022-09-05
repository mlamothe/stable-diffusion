import os
import pickle
import threading
from my_gui.app_options import AppOptions
from my_gui.txt2imglib import load_model, generate
from tqdm import trange
from tkinter import *
from tkinter.ttk import *


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


def save_opts(options_path_file, opts):
    with open(options_path_file, "wb") as f_options:
        pickle.dump(opts, f_options)


def get_opts(options_path_file):
    if not os.path.exists(options_path_file):
        opts = AppOptions()
        save_opts(options_path_file, opts)
    else:
        with open(options_path_file, "rb") as f_options:
            opts = pickle.load(f_options)

    return opts


def append_to_history_file(history_path_file, image_file_name, opts):
    with open(history_path_file, "at") as f_history:
        print(
            f"{image_file_name}; Seed={opts.seed}; Steps={opts.ddim_steps}; Scale={opts.scale}; WxH={opts.W}x{opts.H}; Prompt={opts.prompt}; C={opts.C}; f={opts.f}; ddim_eta={opts.ddim_eta}; precision={opts.precision}; plms={opts.plms}; fixed_code={opts.fixed_code}",
            file=f_history,
        )


class App:
    model: any
    history_path_file: str
    options_path_file: str
    opts: AppOptions
    btn_generate: Button
    default_steps_var: str
    default_seed_var: str
    default_iter_var: str
    default_scale_var: str

    def generate(self):
        self.btn_generate.config(state=DISABLED)

        self.opts.prompt = self.txt_prompt.get("1.0", END).strip()
        self.opts.ddim_steps = int(self.spin_steps.get())
        self.opts.seed = int(self.spin_seed.get())
        self.opts.n_iter = int(self.spin_iter.get())
        self.opts.scale = float(self.scale_scale.get())
        # todo: add more options
        save_opts(self.options_path_file, self.opts)

        for _ in trange(self.opts.n_iter, desc="Sampling"):
            image_file_name = generate(self.model, self.opts)
            append_to_history_file(self.history_path_file, image_file_name, self.opts)
            self.opts.seed += 1
        self.btn_generate.config(state=NORMAL)

    def create_gui(self, opts: AppOptions):
        self.window = Tk()
        self.window.title('My Stable Diffusion')
        self.window.eval("tk::PlaceWindow . center")

        self.btn_generate = Button(
            text="Loading...", command=self.generate, state=DISABLED
        )
        self.btn_generate.pack()

        # TODO: https://stackoverflow.com/questions/40617515/python-tkinter-text-modified-callback
        #       Respond to "change" events
        Label(text="Prompt:").pack()
        self.txt_prompt = Text()
        self.txt_prompt.insert(INSERT, opts.prompt)
        self.txt_prompt.pack()

        Label(text="Steps:").pack()
        self.default_steps_var = StringVar(value=opts.ddim_steps)
        self.spin_steps = Spinbox(from_=1, to=300, textvariable=self.default_steps_var)
        self.spin_steps.pack()

        Label(text="Seed:").pack()
        self.default_seed_var = StringVar(value=opts.seed)
        self.spin_seed = Spinbox(
            from_=1, to=2147483647, textvariable=self.default_seed_var
        )
        self.spin_seed.pack()

        Label(text="# of Iterations:").pack()
        self.default_iter_var = StringVar(value=opts.n_iter)
        self.spin_iter = Spinbox(from_=1, to=10000, textvariable=self.default_iter_var)
        self.spin_iter.pack()

        Label(text="Scale:").pack()
        self.default_scale_var = StringVar(value=opts.scale)
        self.scale_scale = Spinbox(
            from_=1.0, to=100.0, textvariable=self.default_scale_var
        )
        self.scale_scale.pack()

        # todo: cancel button

        return self.window

    def load_model(self):
        self.model = load_model(self.opts)
        self.btn_generate.config(state=NORMAL)
        self.btn_generate.config(text="Generate")
        self.opts.n_iter = 2

        self.history_path_file = prep_history_file()

    def start(self):
        self.options_path_file = prep_options_file()
        self.opts = get_opts(self.options_path_file)

        threading.Thread(target=self.load_model).start()

        window = self.create_gui(self.opts)
        window.mainloop()


def main():
    App().start()


if __name__ == "__main__":
    main()
