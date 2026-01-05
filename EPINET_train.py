"""
Updated for TensorFlow 2.x / Keras 3 compatibility
Original Author: shinyonsei2
Refactored for modularity and readability.
"""

import datetime
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

from epinet_fun.func_epinetmodel import define_epinet
from epinet_fun.func_generate_traindata import (
    data_augmentation_for_train,
    generate_traindata512,
    generate_traindata_for_train,
)
from epinet_fun.func_savedata import display_current_output
from epinet_fun.util import load_LFdata


# --- Configuration ---
@dataclass
class TrainingConfig:
    # Model Hyperparameters
    network_name: str = "epinet_validation"
    model_conv_depth: int = 7
    model_filt_num: int = 70
    learning_rate: float = 0.1e-3
    batch_size: int = 32
    epochs: int = 3

    # Data Parameters
    input_size: int = 25  # 23 + 2
    image_width: int = 512
    image_height: int = 512
    angular_views: np.ndarray = field(
        default_factory=lambda: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    )

    # Training Settings
    display_status_ratio: int = 10000
    load_weights: bool = False

    # Directories
    base_ckpt_dir: str = "epinet_checkpoints"
    base_out_dir: str = "epinet_output"

    # Dataset Paths
    train_dirs: List[str] = field(
        default_factory=lambda: [
            "additional/antinous",
            "additional/boardgames",
            "additional/dishes",
            "additional/greek",
            "additional/kitchen",
            "additional/medieval2",
            "additional/museum",
            "additional/pens",
            "additional/pillows",
            "additional/platonic",
            "additional/rosemary",
            "additional/table",
            "additional/tomb",
            "additional/tower",
            "additional/town",
            "additional/vinyl",
        ]
    )

    val_dirs: List[str] = field(
        default_factory=lambda: [
            "stratified/backgammon",
            "stratified/dots",
            "stratified/pyramids",
            "stratified/stripes",
            "training/boxes",
            "training/cotton",
            "training/dino",
            "training/sideboard",
        ]
    )

    @property
    def label_size(self):
        return self.input_size - 22

    @property
    def ckpt_dir(self):
        return os.path.join(self.base_ckpt_dir, f"{self.network_name}_ckp")

    @property
    def output_dir(self):
        return os.path.join(self.base_out_dir, self.network_name)

    @property
    def log_file(self):
        return os.path.join(self.base_ckpt_dir, f"lf_{self.network_name}.txt")


# --- Generators ---
def training_generator(traindata_all, traindata_label, config: TrainingConfig):
    """
    Yields batches of training data with augmentation.
    """
    while True:
        # Generate basic batch
        (batch_90d, batch_0d, batch_45d, batch_m45d, label_batch) = (
            generate_traindata_for_train(
                traindata_all,
                traindata_label,
                config.input_size,
                config.label_size,
                config.batch_size,
                config.angular_views,
            )
        )

        # Apply augmentation
        (batch_90d, batch_0d, batch_45d, batch_m45d, label_batch) = (
            data_augmentation_for_train(
                batch_90d,
                batch_0d,
                batch_45d,
                batch_m45d,
                label_batch,
                config.batch_size,
            )
        )

        # Expand dims for channel
        label_batch = label_batch[:, :, :, np.newaxis]

        yield (
            (batch_90d, batch_0d, batch_45d, batch_m45d),
            label_batch,
        )


# --- Helper Functions ---
def setup_directories(config: TrainingConfig):
    """Creates necessary directories for checkpoints and outputs."""
    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize log file with timestamp
    with open(config.log_file, "a") as f:
        f.write(f"\n{datetime.datetime.now()}\n\n")


def load_and_prepare_data(config: TrainingConfig):
    """Loads training and validation datasets."""
    print("Loading training data...")
    train_all, train_label = load_LFdata(config.train_dirs)

    # Pre-generate full views for validation/prediction later
    train_90d, train_0d, train_45d, train_m45d, _ = generate_traindata512(
        train_all, train_label, config.angular_views
    )
    print("Loading training data... Complete")

    print("Loading test data...")
    val_all, val_label = load_LFdata(config.val_dirs)
    # Note: validation views are loaded but not explicitly returned here
    # as the original script uses 'traindata_*' for the intermediate prediction step.
    print("Loading test data... Complete")

    return (train_all, train_label), (train_90d, train_0d, train_45d, train_m45d)


def initialize_models(config: TrainingConfig):
    """Defines the patch-based training model and full-image prediction model."""
    model_train = define_epinet(
        config.input_size,
        config.input_size,
        config.angular_views,
        config.model_conv_depth,
        config.model_filt_num,
        config.learning_rate,
    )

    model_predict = define_epinet(
        config.image_width,
        config.image_height,
        config.angular_views,
        config.model_conv_depth,
        config.model_filt_num,
        config.learning_rate,
    )
    return model_train, model_predict


def manage_checkpoints(model, config: TrainingConfig) -> int:
    """Loads the latest checkpoint if configured and returns the starting iteration."""
    start_iter = 0
    if config.load_weights:
        files = os.listdir(config.ckpt_dir)
        checkpoints = []
        valid_files = []

        for f in files:
            if f == "checkpoint":
                continue
            try:
                # Expected format: iterXXXX_...
                iter_num = int(f.split("_")[0][4:])
                checkpoints.append(iter_num)
                valid_files.append(f)
            except ValueError:
                continue

        if checkpoints:
            best_idx = np.argmax(checkpoints)
            start_iter = checkpoints[best_idx] + 1
            ckpt_name = valid_files[best_idx]

            weights_path = os.path.join(config.ckpt_dir, ckpt_name)
            model.load_weights(weights_path)
            print(f"Network weights loaded from {ckpt_name}")

    return start_iter


# --- Main Loop ---
def train_loop(
    models,
    data_gen,
    validation_views,
    train_labels_full,
    config: TrainingConfig,
    start_iter: int,
):
    model_train, model_predict = models
    train_90d, train_0d, train_45d, train_m45d = validation_views

    current_iter = start_iter
    best_bad_pixel = 100.0

    for _ in range(config.epochs):
        print(f"Start training epoch/iteration: {current_iter}")

        # 1. Train on patches
        model_train.fit(
            data_gen,
            steps_per_epoch=int(config.display_status_ratio),
            epochs=current_iter + 1,
            initial_epoch=current_iter,
            verbose=1,
        )

        current_iter += 1

        # 2. Validation / Visualization
        # Transfer weights to full-size model
        model_predict.set_weights(model_train.get_weights())

        # Predict on full images (batch_size=1)
        pred_output = model_predict.predict(
            [train_90d, train_0d, train_45d, train_m45d],
            batch_size=1,
            verbose=0,
        )

        # Calculate metrics and save image
        train_error, train_bp = display_current_output(
            pred_output, train_labels_full, current_iter, config.output_dir
        )

        mse_score = 100 * np.average(np.square(train_error))
        bp_score = 100 * np.average(train_bp)

        print(f"Validation: BP={bp_score:.2f}, MSE={mse_score:.3f}")

        # 3. Save Checkpoints
        save_filename = (
            f"iter{current_iter:04d}_trainmse{mse_score:.3f}_bp{bp_score:.2f}.keras"
        )
        save_path = os.path.join(config.ckpt_dir, save_filename)

        # Log to text file
        with open(config.log_file, "a") as f:
            f.write(f".{save_path}\n")

        # Save best model
        if bp_score < best_bad_pixel:
            best_bad_pixel = bp_score
            model_train.save(save_path)
            print("Model Saved! Best Bad Pixel Ratio so far.")


def main():
    # 1. Setup
    config = TrainingConfig()
    setup_directories(config)

    # 2. Data Loading
    (train_all, train_label), validation_views = load_and_prepare_data(config)

    # 3. Model Initialization
    model_train, model_predict = initialize_models(config)
    start_iter = manage_checkpoints(model_train, config)

    # 4. Generator Setup
    gen_instance = training_generator(train_all, train_label, config)

    # 5. Run Training
    train_loop(
        models=(model_train, model_predict),
        data_gen=gen_instance,
        validation_views=validation_views,
        train_labels_full=train_label,
        config=config,
        start_iter=start_iter,
    )


if __name__ == "__main__":
    main()
