import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

# Suppress future warnings from XGBRegressor
warnings.filterwarnings("ignore", category=FutureWarning)

class MLApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Views Prediction Model")
        self.root.geometry("1200x800")

        self.train_data = None
        self.test_data = None
        self.model = None
        self.output_queue = queue.Queue()

        self.create_gui()
        self.setup_output_handling()

    def create_gui(self):
        self.control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        self.control_frame.pack(fill="x", padx=10, pady=5)

        self.output_frame = ttk.LabelFrame(self.root, text="Output", padding="10")
        self.output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.create_control_elements()
        self.create_output_elements()

    def create_control_elements(self):
        ttk.Button(self.control_frame, text="Upload Training Data",
                   command=self.load_training_data).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Upload Test Data",
                   command=self.load_test_data).pack(side="left", padx=5)

        self.train_button = ttk.Button(self.control_frame, text="Train Model",
                                       command=self.start_training, state="disabled")
        self.train_button.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(self.control_frame, length=200, mode='indeterminate')
        self.progress.pack(side="left", padx=5)

    def create_output_elements(self):
        self.notebook = ttk.Notebook(self.output_frame)
        self.notebook.pack(fill="both", expand=True)

        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Log")

        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=20)
        self.log_text.pack(fill="both", expand=True)

        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualizations")

    def setup_output_handling(self):
        def check_output():
            try:
                while True:
                    message = self.output_queue.get_nowait()
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
            except queue.Empty:
                pass
            self.root.after(100, check_output)

        self.root.after(100, check_output)

    def load_training_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.train_data = pd.read_csv(file_path)
                self.output_queue.put(f"Training data loaded: {file_path}")
                self.output_queue.put(f"Training data shape: {self.train_data.shape}")
                self.check_enable_training()
            except Exception as e:
                self.output_queue.put(f"Error loading training data: {str(e)}")

    def load_test_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.test_data = pd.read_csv(file_path)
                self.output_queue.put(f"Test data loaded: {file_path}")
                self.output_queue.put(f"Test data shape: {self.test_data.shape}")
                self.check_enable_training()
            except Exception as e:
                self.output_queue.put(f"Error loading test data: {str(e)}")

    def check_enable_training(self):
        if self.train_data is not None and self.test_data is not None:
            self.train_button.config(state="normal")

    def clean_data(self, df):
        df = df.copy()
        df['Average view duration'] = df['Average view duration'].apply(
            lambda x: self.convert_duration_to_seconds(x))
        return df.dropna()

    def convert_duration_to_seconds(self, duration):
        if pd.isna(duration) or not isinstance(duration, str):
            return np.nan
        try:
            return sum(int(i) * 60 ** idx for idx, i in enumerate(reversed(duration.split(':'))))
        except ValueError:
            return np.nan

    def prepare_data(self, df):
        feature_columns = ['Shares', 'Comments added', 'Likes (vs. dislikes) (%)',
                           'Average view duration', 'Subscribers',
                           'Impressions click-through rate (%)']
        df = df.fillna(0)  # Handle remaining NaN values by filling with 0
        X = df[feature_columns]
        y = df['Views']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.output_queue.put("Training XGBoost model...")

        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_val, y_val):
        y_pred = model.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)

        self.output_queue.put("\nModel Performance Metrics:")
        self.output_queue.put(f"RMSE: {rmse:,.2f}")
        self.output_queue.put(f"R² Score: {r2:.4f}")

        return y_pred

    def feature_importance_plot(self, model, feature_columns):
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        self.output_queue.put("\nFeature Importance:")
        self.output_queue.put(str(feature_importance))

        self.create_feature_importance_plot(feature_importance)

    def create_feature_importance_plot(self, feature_importance_df):
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance_df.plot(kind='bar', x='Feature', y='Importance', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def start_training(self):
        self.train_button.config(state="disabled")
        self.progress.start()

        def train_thread():
            try:
                self.output_queue.put("Starting data preprocessing...")

                train_clean = self.clean_data(self.train_data)
                X_train, X_val, y_train, y_val = self.prepare_data(train_clean)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                self.model = self.train_model(X_train_scaled, y_train)
                self.evaluate_model(self.model, X_val_scaled, y_val)
                self.feature_importance_plot(self.model, X_train.columns)

                self.output_queue.put("\nModel training completed successfully!")
            except Exception as e:
                self.output_queue.put(f"Error during training: {str(e)}")
            finally:
                self.progress.stop()
                self.train_button.config(state="normal")

        threading.Thread(target=train_thread, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApplication(root)
    root.mainloop()
