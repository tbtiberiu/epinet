import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

# --- Project Imports ---
try:
    from utils.func_epinetmodel import define_epinet
    from utils.func_makeinput import make_multiinput
except ImportError as e:
    print(f'Import Error: {e}')
    print(
        "Please ensure you are running this script from the project root and the 'utils' folder exists."
    )
    exit(1)


class EPINETDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title('EPINET Depth Estimation - Interactive Demo')
        self.root.geometry('1200x800')

        # --- Internal State ---
        self.input_dir = None
        self.model = None
        self.current_h = 512
        self.current_w = 512
        self.current_views = []

        # --- Default Configuration Variables ---
        self.var_weights_path = tk.StringVar(
            value='models/checkpoints/iter0002_trainmse9.728_bp38.80.keras'
        )
        self.var_dataset_type = tk.StringVar(
            value='synthetic'
        )  # Options: synthetic, lytro
        self.var_view_mode = tk.StringVar(value='9x9')  # Options: 9x9, 5x5

        self.setup_ui()

    def setup_ui(self):
        # =========================================================================
        # 1. Configuration Panel (Top)
        # =========================================================================
        config_frame = tk.LabelFrame(
            self.root,
            text='1. Model Configuration',
            padx=10,
            pady=10,
            font=('Arial', 10, 'bold'),
        )
        config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Row A: Weights File Selection
        frame_weights = tk.Frame(config_frame)
        frame_weights.pack(fill=tk.X, pady=2)
        tk.Label(frame_weights, text='Weights Path:', width=15, anchor='w').pack(
            side=tk.LEFT
        )
        tk.Entry(frame_weights, textvariable=self.var_weights_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )
        tk.Button(frame_weights, text='Browse...', command=self.browse_weights).pack(
            side=tk.LEFT
        )

        # Row B: Dataset Type & View Mode
        frame_opts = tk.Frame(config_frame)
        frame_opts.pack(fill=tk.X, pady=5)

        # Dataset Type
        tk.Label(frame_opts, text='Dataset Type:', width=15, anchor='w').pack(
            side=tk.LEFT
        )
        tk.Radiobutton(
            frame_opts,
            text='Synthetic (512x512)',
            variable=self.var_dataset_type,
            value='synthetic',
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            frame_opts,
            text='Lytro (552x383)',
            variable=self.var_dataset_type,
            value='lytro',
        ).pack(side=tk.LEFT, padx=10)

        # Separator
        tk.Frame(frame_opts, width=2, bg='gray').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # View Mode
        tk.Label(frame_opts, text='View Mode:', width=10, anchor='w').pack(side=tk.LEFT)
        tk.Radiobutton(
            frame_opts, text='9x9 Views', variable=self.var_view_mode, value='9x9'
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            frame_opts, text='5x5 Views', variable=self.var_view_mode, value='5x5'
        ).pack(side=tk.LEFT)

        # Row C: Load Model Button
        self.btn_load_model = tk.Button(
            config_frame,
            text='Load / Build Model',
            command=self.load_model,
            bg='#2196F3',
            fg='white',
            font=('Arial', 9, 'bold'),
        )
        self.btn_load_model.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # =========================================================================
        # 2. Input & Execution (Middle Top)
        # =========================================================================
        exec_frame = tk.LabelFrame(
            self.root,
            text='2. Input & Execution',
            padx=10,
            pady=10,
            font=('Arial', 10, 'bold'),
        )
        exec_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.btn_browse_folder = tk.Button(
            exec_frame, text='Select Image Folder', command=self.select_folder, height=2
        )
        self.btn_browse_folder.pack(side=tk.LEFT, padx=5)

        self.lbl_path = tk.Label(
            exec_frame,
            text='No folder selected',
            fg='gray',
            relief=tk.SUNKEN,
            anchor='w',
        )
        self.lbl_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.btn_run = tk.Button(
            exec_frame,
            text='RUN ESTIMATION',
            command=self.run_inference,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            height=2,
            padx=20,
        )
        self.btn_run.pack(side=tk.RIGHT, padx=5)
        self.btn_run['state'] = (
            'disabled'  # Disabled until model is loaded and folder selected
        )

        # =========================================================================
        # 3. Visualization (Center)
        # =========================================================================
        display_frame = tk.Frame(self.root)
        display_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        # Left Panel (Input)
        self.panel_left = tk.LabelFrame(display_frame, text='Center Input View')
        self.panel_left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        self.canvas_input = tk.Label(
            self.panel_left, text='Waiting for input...', bg='#f0f0f0'
        )
        self.canvas_input.pack(expand=True, fill=tk.BOTH)

        # Right Panel (Output)
        self.panel_right = tk.LabelFrame(
            display_frame, text='Estimated Depth Map (Disparity)'
        )
        self.panel_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)
        self.canvas_output = tk.Label(
            self.panel_right, text='Waiting for results...', bg='#f0f0f0'
        )
        self.canvas_output.pack(expand=True, fill=tk.BOTH)

        # =========================================================================
        # 4. Status Bar (Bottom)
        # =========================================================================
        self.status_bar = tk.Label(
            self.root,
            text='Welcome. Please configure and load the model.',
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Logic Methods ---

    def log(self, message):
        self.status_bar.config(text=f'Status: {message}')
        self.root.update_idletasks()

    def browse_weights(self):
        filename = filedialog.askopenfilename(
            filetypes=[('Keras/HDF5 Files', '*.keras *.hdf5 *.h5')]
        )
        if filename:
            self.var_weights_path.set(filename)

    def load_model(self):
        """
        Builds the model based on the selected radio buttons (dimensions and views).
        """
        weights_path = self.var_weights_path.get()
        ds_type = self.var_dataset_type.get()
        v_mode = self.var_view_mode.get()

        if not os.path.exists(weights_path):
            messagebox.showerror(
                'File Error', f'Weights file not found:\n{weights_path}'
            )
            return

        # 1. Determine Dimensions
        if ds_type == 'synthetic':
            self.current_h = 512
            self.current_w = 512
        elif ds_type == 'lytro':
            self.current_h = 383
            self.current_w = 552

        # 2. Determine Views
        if v_mode == '9x9':
            self.current_views = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            self.current_views = [2, 3, 4, 5, 6]  # 5x5

        self.log(f'Building model ({ds_type}, {v_mode})... Please wait.')
        self.root.config(cursor='watch')

        try:
            # Re-define model
            self.model = define_epinet(
                sz_input_h=self.current_h,
                sz_input_w=self.current_w,
                view_n=self.current_views,
                conv_depth=7,
                filt_num=70,
                learning_rate=0.1e-3,
            )

            self.log(f'Loading weights from {os.path.basename(weights_path)}...')
            self.model.load_weights(weights_path)

            # Run a dummy prediction to initialize the graph
            self.log('Warming up GPU/CPU...')
            dum_sz = self.model.input_shape[0]  # Get shape from first input
            # Create 4 dummy inputs (90d, 0d, 45d, -45d)
            dum = np.zeros((1, dum_sz[1], dum_sz[2], dum_sz[3]), dtype=np.float32)
            self.model.predict([dum, dum, dum, dum], batch_size=1, verbose=0)

            self.log('Model Loaded Successfully!')
            messagebox.showinfo('Success', 'Model loaded and ready.')

            if self.input_dir:
                self.btn_run['state'] = 'normal'

        except Exception as e:
            self.log(f'Error loading model: {e}')
            messagebox.showerror('Model Error', str(e))
            self.model = None
            self.btn_run['state'] = 'disabled'
        finally:
            self.root.config(cursor='')

    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.input_dir = folder_selected
            self.lbl_path.config(text=folder_selected)
            self.show_preview_image(folder_selected)
            self.log(f'Folder selected: {os.path.basename(folder_selected)}')

            if self.model is not None:
                self.btn_run['state'] = 'normal'
            else:
                self.log('Please load the model before running.')

    def show_preview_image(self, folder):
        try:
            # Try to find a central image.
            # For 9x9, Cam040 is center. For Lytro, it varies, but we take a guess.
            preview_path = None

            # Scan directory
            files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])

            if not files:
                self.canvas_input.config(text='No .png files found', image='')
                return

            # Logic to find "middle" file
            if 'input_Cam040.png' in files:
                preview_path = os.path.join(folder, 'input_Cam040.png')
            else:
                # Pick the middle file in the list
                preview_path = os.path.join(folder, files[len(files) // 2])

            if preview_path:
                img = Image.open(preview_path)
                img.thumbnail((450, 450))
                photo = ImageTk.PhotoImage(img)
                self.canvas_input.config(image=photo, text='')
                self.canvas_input.image = photo
        except Exception as e:
            print(f'Preview error: {e}')

    def run_inference(self):
        if not self.model or not self.input_dir:
            return

        self.log('Preprocessing inputs...')
        self.root.config(cursor='watch')
        self.btn_run['state'] = 'disabled'

        try:
            # Call the updated make_multiinput (handles absolute paths)
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

            # --- Visualization ---
            disparity_map = pred[0, :, :, 0]

            # Normalize for display
            d_min, d_max = disparity_map.min(), disparity_map.max()
            disp_norm = (disparity_map - d_min) / (d_max - d_min + 1e-8)

            # Apply Colormap
            colormap = plt.get_cmap('plasma')
            disp_colored = (colormap(disp_norm)[:, :, :3] * 255).astype(np.uint8)

            # Update UI
            res_img = Image.fromarray(disp_colored)
            res_img.thumbnail((450, 450))
            photo_res = ImageTk.PhotoImage(res_img)

            self.canvas_output.config(image=photo_res, text='')
            self.canvas_output.image = photo_res

            save_dir = 'results'
            Path(save_dir).mkdir(exist_ok=True)
            folder_name = os.path.basename(os.path.normpath(self.input_dir))
            save_path = os.path.join(save_dir, f'{folder_name}_depth.png')
            Image.fromarray(disp_colored).save(save_path)
            self.log(f'Saved visualization to {save_path}')

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
