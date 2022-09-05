from tkinter import *
from tkinter.ttk import *
from my_gui.app_options import AppOptions


class Gui:
    DIMENSIONS = [
        "448x512",
        "448x640",
        "512x448",
        "512x512",
        "640x448",
    ]

    _btn_generate: Button
    _txt_prompt: Text
    _spin_steps: Spinbox
    _spin_seed: Spinbox
    _spin_iter: Spinbox
    _spin_scale: Spinbox
    _chk_increment_scale: Checkbutton
    _opt_dimension: OptionMenu
    _chk_super_randomize: Checkbutton

    _dimension_var: int
    _default_steps_var: int
    _default_seed_var: int
    _default_iter_var: int
    _default_scale_var: float
    _super_randomize_var: int

    def dimensions(self):
        return self.DIMENSIONS

    def create_gui(self, opt: AppOptions, generate_func):
        window = Tk()
        window.title("My Stable Diffusion")
        window.eval("tk::PlaceWindow . center")

        self._btn_generate = Button(
            text="Loading...", command=generate_func, state=DISABLED
        )
        self._btn_generate.pack()

        # TODO: https://stackoverflow.com/questions/40617515/python-tkinter-text-modified-callback
        #       Respond to "change" events
        Label(text="Prompt:").pack()
        self._txt_prompt = Text()
        self._txt_prompt.insert(INSERT, opt.prompt)
        self._txt_prompt.pack()

        Label(text="Dimensions:").pack()
        default_dimension = f"{opt.W}x{opt.H}"
        self._dimension_var = StringVar(value=f"{opt.W}x{opt.H}")
        self._opt_dimension = OptionMenu(
            window, self._dimension_var, default_dimension, *self.DIMENSIONS
        )
        self._opt_dimension.pack()

        Label(text="Steps:").pack()
        self._default_steps_var = StringVar(value=opt.ddim_steps)
        self._spin_steps = Spinbox(
            from_=1, to=300, textvariable=self._default_steps_var
        )
        self._spin_steps.pack()

        Label(text="Seed:").pack()
        self._default_seed_var = IntVar(value=opt.seed)
        self._spin_seed = Spinbox(
            from_=1, to=2147483647, textvariable=self._default_seed_var
        )
        self._spin_seed.pack()

        Label(text="Scale:").pack()
        self._default_scale_var = DoubleVar(value=opt.scale)
        self._spin_scale = Spinbox(
            from_=1.0, to=100.0, textvariable=self._default_scale_var
        )
        self._spin_scale.pack()

        Label(text="# of Iterations:").pack()
        self._default_iter_var = IntVar(value=opt.n_iter)
        self._spin_iter = Spinbox(
            from_=1, to=10000, textvariable=self._default_iter_var
        )
        self._spin_iter.pack()

        self._mov_file_names = IntVar(value=opt.mov_file_names)
        self._chk_mov_file_names = Checkbutton(
            text="Movie File Names",
            variable=self._mov_file_names,
            onvalue=1,
            offvalue=0,
        )
        self._chk_mov_file_names.pack()

        self._increment_scale_var = IntVar(value=opt.increment_scale)
        self._chk_increment_scale = Checkbutton(
            text="Increment Scale by 0.5",
            variable=self._increment_scale_var,
            onvalue=1,
            offvalue=0,
        )
        self._chk_increment_scale.pack()

        self._super_randomize_var = IntVar(value=opt.super_randomize)
        self._chk_super_randomize = Checkbutton(
            text="Super Randomize!",
            variable=self._super_randomize_var,
            onvalue=1,
            offvalue=0,
        )
        self._chk_super_randomize.pack()

        # todo: cancel button

        window.mainloop()

    def get_opts(self, opt):
        opt.prompt = self._txt_prompt.get("1.0", END).strip()
        opt.ddim_steps = int(self._spin_steps.get())
        opt.seed = int(self._spin_seed.get())
        opt.n_iter = int(self._spin_iter.get())
        opt.scale = float(self._spin_scale.get())
        opt.mov_file_names = self._mov_file_names.get()
        opt.increment_scale = self._increment_scale_var.get()
        opt.W, opt.H = map(int, self._dimension_var.get().split("x"))
        opt.super_randomize = self._super_randomize_var.get()
        # todo: add more options
        return opt

    def model_loaded(self):
        self._btn_generate.config(text="Generate")
        self._btn_generate.config(state=NORMAL)

    def generating_started(self):
        self._btn_generate.config(state=DISABLED)

    def generating_completed(self):
        self._btn_generate.config(state=NORMAL)
