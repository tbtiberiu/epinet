import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

from utils.func_epinetmodel import define_epinet
from utils.func_makeinput import make_multiinput

# --- Configuration Constants ---
CONFIG = {
    'WINDOW_SIZE': '1200x800',
    'DEFAULT_WEIGHTS': 'models/checkpoints/iter0002_trainmse9.728_bp38.80.keras',
    'COLORS': {
        'BG_LIGHT': '#f0f0f0',
        'BTN_BLUE': '#2196F3',
        'BTN_GREEN': '#4CAF50',
    },
    'THUMBNAIL_SIZE': (450, 450),
}


class EPINETDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title('EPINET Depth Estimation - Interactive Demo')
        self.root.geometry(CONFIG['WINDOW_SIZE'])

        # --- Internal State ---
        self.input_dir = None
        self.model = None
        self.current_h = 512
        self.current_w = 512
        self.current_views = []

        # --- UI Variables ---
        self.var_weights_path = tk.StringVar(value=CONFIG['DEFAULT_WEIGHTS'])
        self.var_dataset_type = tk.StringVar(value='synthetic')
        self.var_view_mode = tk.StringVar(value='9x9')

        self.setup_ui()

    # =========================================================================
    # UI Setup Methods
    # =========================================================================
    def setup_ui(self):
        self._setup_config_panel()
        self._setup_exec_panel()
        self._setup_display_panel()
        self._setup_status_bar()

    def _setup_config_panel(self):
        frame = tk.LabelFrame(
            self.root,
            text='1. Model Configuration',
            padx=10,
            pady=10,
            font=('Arial', 10, 'bold'),
        )
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Row A: Weights
        f_weights = tk.Frame(frame)
        f_weights.pack(fill=tk.X, pady=2)
        tk.Label(f_weights, text='Weights Path:', width=15, anchor='w').pack(
            side=tk.LEFT
        )
        tk.Entry(f_weights, textvariable=self.var_weights_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )
        tk.Button(f_weights, text='Browse...', command=self.browse_weights).pack(
            side=tk.LEFT
        )

        # Row B: Options
        f_opts = tk.Frame(frame)
        f_opts.pack(fill=tk.X, pady=5)

        tk.Label(f_opts, text='Dataset Type:', width=15, anchor='w').pack(side=tk.LEFT)
        for text, val in [
            ('Synthetic (512x512)', 'synthetic'),
            ('Lytro (552x383)', 'lytro'),
        ]:
            tk.Radiobutton(
                f_opts, text=text, variable=self.var_dataset_type, value=val
            ).pack(side=tk.LEFT, padx=5)

        tk.Frame(f_opts, width=2, bg='gray').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        tk.Label(f_opts, text='View Mode:', width=10, anchor='w').pack(side=tk.LEFT)
        for text, val in [('9x9 Views', '9x9'), ('5x5 Views', '5x5')]:
            tk.Radiobutton(
                f_opts, text=text, variable=self.var_view_mode, value=val
            ).pack(side=tk.LEFT)

        # Row C: Load Button
        self.btn_load_model = tk.Button(
            frame,
            text='Load Model',
            command=self.load_model,
            bg=CONFIG['COLORS']['BTN_BLUE'],
            fg='white',
            font=('Arial', 9, 'bold'),
        )
        self.btn_load_model.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def _setup_exec_panel(self):
        frame = tk.LabelFrame(
            self.root,
            text='2. Input & Execution',
            padx=10,
            pady=10,
            font=('Arial', 10, 'bold'),
        )
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Button(
            frame, text='Select Image Folder', command=self.select_folder, height=2
        ).pack(side=tk.LEFT, padx=5)

        self.lbl_path = tk.Label(
            frame, text='No folder selected', fg='gray', relief=tk.SUNKEN, anchor='w'
        )
        self.lbl_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.btn_run = tk.Button(
            frame,
            text='RUN ESTIMATION',
            command=self.run_inference,
            bg=CONFIG['COLORS']['BTN_GREEN'],
            fg='white',
            font=('Arial', 12, 'bold'),
            height=2,
            padx=20,
            state='disabled',
        )
        self.btn_run.pack(side=tk.RIGHT, padx=5)

    def _setup_display_panel(self):
        frame = tk.Frame(self.root)
        frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        # Helper to create panels
        def create_panel(parent, title, side):
            lf = tk.LabelFrame(parent, text=title)
            lf.pack(side=side, expand=True, fill=tk.BOTH, padx=5)
            lbl = tk.Label(lf, text='Waiting...', bg=CONFIG['COLORS']['BG_LIGHT'])
            lbl.pack(expand=True, fill=tk.BOTH)
            return lbl

        self.canvas_input = create_panel(frame, 'Center Input View', tk.LEFT)
        self.canvas_output = create_panel(frame, 'Estimated Depth Map', tk.RIGHT)

    def _setup_status_bar(self):
        self.status_bar = tk.Label(
            self.root,
            text='Welcome. Please configure and load the model.',
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # =========================================================================
    # Logic & Helpers
    # =========================================================================
    def log(self, message):
        self.status_bar.config(text=f'Status: {message}')
        self.root.update_idletasks()

    def browse_weights(self):
        filename = filedialog.askopenfilename(
            filetypes=[('Keras/HDF5 Files', '*.keras *.hdf5 *.h5')]
        )
        if filename:
            self.var_weights_path.set(filename)

    def _get_model_settings(self):
        """Derive model parameters from UI variables."""
        h, w = (512, 512) if self.var_dataset_type.get() == 'synthetic' else (383, 552)
        views = (
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
            if self.var_view_mode.get() == '9x9'
            else [2, 3, 4, 5, 6]
        )
        return h, w, views

    def _find_preview_path(self, folder):
        """Identify a suitable preview image in the folder."""
        files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
        if not files:
            return None
        if 'input_Cam040.png' in files:
            return os.path.join(folder, 'input_Cam040.png')
        return os.path.join(folder, files[len(files) // 2])

    def _display_image_on_canvas(self, image_source, canvas):
        """Resize and display an image (path or array) on a Tkinter canvas."""
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
        elif isinstance(image_source, np.ndarray):
            img = Image.fromarray(image_source)
        else:
            return

        img.thumbnail(CONFIG['THUMBNAIL_SIZE'])
        photo = ImageTk.PhotoImage(img)
        canvas.config(image=photo, text='')
        canvas.image = photo  # Keep reference

    # =========================================================================
    # Main Actions
    # =========================================================================
    def load_model(self):
        weights_path = self.var_weights_path.get()
        if not os.path.exists(weights_path):
            messagebox.showerror('File Error', f'Weights not found:\n{weights_path}')
            return

        self.current_h, self.current_w, self.current_views = self._get_model_settings()

        self.log(
            f'Building model ({self.var_dataset_type.get()}, {self.var_view_mode.get()})...'
        )
        self.root.config(cursor='watch')

        try:
            self.model = define_epinet(
                sz_input_h=self.current_h,
                sz_input_w=self.current_w,
                view_n=self.current_views,
                conv_depth=7,
                filt_num=70,
                learning_rate=0.1e-3,
            )

            self.log('Loading weights...')
            self.model.load_weights(weights_path)

            self.log('Warming up...')
            # Dummy prediction
            dum_shape = (
                1,
                self.model.input_shape[0][1],
                self.model.input_shape[0][2],
                self.model.input_shape[0][3],
            )
            dum = np.zeros(dum_shape, dtype=np.float32)
            self.model.predict([dum] * 4, batch_size=1, verbose=0)

            self.log('Model Loaded Successfully!')
            messagebox.showinfo('Success', 'Model loaded and ready.')

            if self.input_dir:
                self.btn_run['state'] = 'normal'

        except Exception as e:
            self.log(f'Error: {e}')
            messagebox.showerror('Model Error', str(e))
            self.model = None
            self.btn_run['state'] = 'disabled'
        finally:
            self.root.config(cursor='')

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_dir = folder
            self.lbl_path.config(text=folder)
            self.log(f'Folder selected: {os.path.basename(folder)}')

            preview_path = self._find_preview_path(folder)
            if preview_path:
                self._display_image_on_canvas(preview_path, self.canvas_input)
            else:
                self.canvas_input.config(text='No .png files found', image='')

            if self.model:
                self.btn_run['state'] = 'normal'
            else:
                self.log('Please build the model first.')

    def run_inference(self):
        if not self.model or not self.input_dir:
            return

        self.log('Preprocessing inputs...')
        self.root.config(cursor='watch')
        self.btn_run['state'] = 'disabled'

        try:
            val_90d, val_0d, val_45d, val_M45d = make_multiinput(
                self.input_dir, self.current_h, self.current_w, self.current_views
            )

            self.log('Running Neural Network...')
            start_t = time.perf_counter()
            pred = self.model.predict(
                [val_90d, val_0d, val_45d, val_M45d], batch_size=1, verbose=0
            )
            elapsed = time.perf_counter() - start_t
            self.log(f'Inference done in {elapsed:.3f}s. Rendering...')

            # Process Output
            disparity = pred[0, :, :, 0]
            d_min, d_max = disparity.min(), disparity.max()
            disp_norm = (disparity - d_min) / (d_max - d_min + 1e-8)

            # Colorize
            colormap = plt.get_cmap('plasma')
            disp_colored = (colormap(disp_norm)[:, :, :3] * 255).astype(np.uint8)

            # Display and Save
            self._display_image_on_canvas(disp_colored, self.canvas_output)

            save_dir = Path('results')
            save_dir.mkdir(exist_ok=True)
            folder_name = Path(self.input_dir).name
            save_path = save_dir / f'{folder_name}_depth.png'

            Image.fromarray(disp_colored).save(save_path)
            self.log(f'Saved result to {save_path}')

        except Exception as e:
            self.log('Runtime Error')
            messagebox.showerror('Error', str(e))
            print(e)
        finally:
            self.root.config(cursor='')
            self.btn_run['state'] = 'normal'


if __name__ == '__main__':
    root = tk.Tk()
    app = EPINETDemoApp(root)
    root.mainloop()
