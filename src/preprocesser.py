import pandas as pd

from config import ProjectConfig


class DataProcessor:
    def __init__(self):
        self.config = ProjectConfig()
        self.data = None

    def load_data(self):
        """Load the raw NBA players data."""
        self.data = pd.read_csv(self.config.RAW_DATA_FILE)
        return self

    def clean_data(self):
        """Perform basic data cleaning operations."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Drop the unnamed index column
        if "Unnamed: 0" in self.data.columns:
            self.data = self.data.drop("Unnamed: 0", axis=1)

        # Handle missing values
        self.data = self.data.fillna(
            {
                "college": "Unknown",
                "country": "Unknown",
                "draft_year": "Undrafted",
                "draft_round": "Undrafted",
                "draft_number": "Undrafted",
            }
        )

        return self

    def process_numerical_features(self):
        """Process numerical features."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Convert heights to meters and weights to kg
        self.data["player_height"] = self.data["player_height"] / 100  # assuming height is in cm

        # Scale numerical features if needed
        # Add more numerical processing as needed

        return self

    def get_processed_data(self):
        """Return the processed dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data
