import pandas as pd

class AnimeDataLoader:
    def __init__(self, file_path: str, processed_file_path: str = None):
        """ Initializes the data loader with the path to the CSV file."""
        self.file_path = file_path
        self.processed_file_path = processed_file_path

    def load_and_process_data(self):
        try:
            df = pd.read_csv(self.file_path, encoding='utf-8', on_bad_lines='skip').dropna()            
            required_columns = {"Name", "Genres", "sypnopsis"}

            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            df['combined_info'] = (
                "Title: " + df['Name'] + "Overview: " + df['sypnopsis'] + "Genres: " + df['Genres'] + "\n"
            )

            df[['combined_info']].to_csv(self.processed_file_path, index=False, encoding='utf-8')

            return self.processed_file_path
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except pd.errors.EmptyDataError:
            print("No data found in the file.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None