from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.filters import threshold_otsu
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import glob
import random


class ICTest:
    def __init__(self, image_path: str):
        """Initialize the MicroscopyImage class and load the image."""
        self.image = Image.open(image_path)
        self.image_array = np.array(self.image)
        self.background = None  # This will be populated in the segmentation step

    def plot_image_histogram(self):
        """Plot the image and its histogram side by side."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Display the image
        ax[0].imshow(self.image_array, cmap="gray")
        ax[0].set_title("Original Image")

        # Display the histogram
        ax[1].hist(self.image_array.ravel(), bins=256, color="gray", alpha=0.7)
        ax[1].set_title("Histogram of Pixel Intensities")

        plt.show()

    def segment_image(self, thresholding_method: str = "otsu"):
        """Perform image segmentation using the provided thresholding method (default is Otsu's method)."""
        if thresholding_method == "otsu":
            thresh = threshold_otsu(self.image_array)
        else:
            raise ValueError(f"Unknown thresholding method: {thresholding_method}")

        binary = self.image_array > thresh
        self.background = np.where(binary, 0, self.image_array)

        # Display the original image, the binary image after thresholding, and the identified background
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].imshow(self.image_array, cmap="gray")
        ax[0].set_title("Original Image")

        ax[1].imshow(binary, cmap="gray")
        ax[1].set_title("Binary Image after Thresholding")

        ax[2].imshow(self.background, cmap="gray")
        ax[2].set_title("Identified Background")

        plt.show()

    def calculate_illumination_score(self):
        """Calculate and return the illumination correction score."""
        if self.background is None:
            raise RuntimeError(
                "You need to segment the image first before calculating the illumination score."
            )

        # Calculate the average pixel intensity for each row (axis=1) and column (axis=0)
        row_averages = np.mean(self.background, axis=1)
        column_averages = np.mean(self.background, axis=0)

        # Calculate the slope for each row and column
        row_slope, _, _, _, _ = stats.linregress(
            np.arange(len(row_averages)), row_averages
        )
        column_slope, _, _, _, _ = stats.linregress(
            np.arange(len(column_averages)), column_averages
        )

        # Calculate the score as the average of the absolute slopes
        score = np.mean([np.abs(row_slope), np.abs(column_slope)])

        return score


class ParallelICTest:
    def __init__(
        self,
        directory: str,
        glob_pattern: str,
        sampling_fraction: float,
        num_workers: int,
    ):
        """Initialize the ParallelICTest class."""
        self.directory = directory
        self.glob_pattern = glob_pattern
        self.sampling_fraction = sampling_fraction
        self.num_workers = num_workers
        self.scores_df = pd.DataFrame(columns=["file_name", "illumination_score"])

        # Get the list of all matching files in the directory
        all_files = glob.glob(f"{directory}/{glob_pattern}")

        # Randomly sample a fraction of these files
        self.sampled_files = random.sample(
            all_files, int(len(all_files) * sampling_fraction)
        )

    def process_image(self, image_path: str):
        """Process a single image and return its illumination score."""
        image = ICTest(image_path)
        image.segment_image()
        score = image.calculate_illumination_score()
        return score

    def process_images(self):
        """Process all sampled images in parallel and store the scores in a DataFrame."""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for image_path, score in zip(
                self.sampled_files, executor.map(self.process_image, self.sampled_files)
            ):
                self.scores_df = self.scores_df.append(
                    {"file_name": image_path, "illumination_score": score},
                    ignore_index=True,
                )

    def get_scores(self):
        """Return the DataFrame with the illumination scores."""
        return self.scores_df
