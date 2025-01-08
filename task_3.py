import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import pandas as pd
import re
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import threading
import openai
import os
from dotenv import load_dotenv
import spacy

class RetentionIntroOptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audience Retention Optimization")
        self.root.geometry("1200x800")

        self.model = None
        self.pipeline = None

        self.create_gui()

    def create_gui(self):
        self.control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        self.control_frame.pack(fill="x", padx=10, pady=5)

        self.output_frame = ttk.LabelFrame(self.root, text="Output", padding="10")
        self.output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.create_control_elements()
        self.create_output_elements()

    def create_control_elements(self):
        ttk.Button(self.control_frame, text="Upload Data Folder",
                   command=self.load_data_folder).pack(side="left", padx=5)

        self.train_button = ttk.Button(self.control_frame, text="Train Model",
                                       command=self.train_model, state="disabled")
        self.train_button.pack(side="left", padx=5)

        ttk.Button(self.control_frame, text="Upload Sample",
                   command=self.optimize_intro_using_reps).pack(side="left", padx=5)

    def create_output_elements(self):
        self.log_text = scrolledtext.ScrolledText(self.output_frame, height=20)
        self.log_text.pack(fill="both", expand=True)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def load_data_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            self.log("No folder selected.")
            return
        self.log(f"Loading data from folder: {folder_path}")
        self.data = []
        for file in os.listdir(folder_path):
            if file.endswith(".docx"):
                base_name = file.replace(".docx", "")
                transcript_path = os.path.join(folder_path, file)
                retention_path = os.path.join(folder_path, base_name + ".xlsx")

                if os.path.exists(retention_path):
                    try:
                        intro_text = self.extract_intro_from_transcript(transcript_path)
                        avg_retention = self.load_retention_data(retention_path)
                        self.data.append({
                            'Introduction': intro_text,
                            'Average Retention': avg_retention
                        })
                    except Exception as e:
                        self.log(f"Error processing {base_name}: {e}")

        self.log(f"Loaded {len(self.data)} video data pairs.")
        if len(self.data) > 0:
            self.train_button.config(state="normal")

    def extract_intro_from_transcript(self, docx_path, sentence_count=3):
        doc = Document(docx_path)
        full_text = " ".join([para.text for para in doc.paragraphs])
        sentences = re.split(r'(?<=[.!?]) +', full_text)
        return " ".join(sentences[:sentence_count])

    def load_retention_data(self, xlsx_path):
        retention_data = pd.read_excel(xlsx_path)
        if 'Absolute audience retention (%)' not in retention_data.columns:
            raise ValueError(f"Expected column 'Absolute audience retention (%)' not found in {xlsx_path}")

        early_positions = retention_data.loc[:10, 'Absolute audience retention (%)']
        early_positions = early_positions.apply(lambda x: x if isinstance(x, (int, float)) else None).dropna()
        return early_positions.mean()

    def train_model(self):
        self.log("Training model...")
        df = pd.DataFrame(self.data)
        X = df['Introduction']
        y = df['Average Retention']

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        self.pipeline.fit(X, y)
        self.log("Model training completed.")

    def optimize_intro_using_reps(self):
        transcript_path = filedialog.askopenfilename(filetypes=[("DOCX files", "*.docx")])
        retention_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])

        if not transcript_path or not retention_path:
            self.log("Please upload both a transcript and retention table.")
            return

        self.clear_log()  # Clear the log for clean output
        loading_label = tk.Label(self.output_frame, text="Processing...", font=("Helvetica", 12, "bold"))
        loading_label.pack()
        self.root.update()

        try:
            original_intro = self.extract_intro_from_transcript(transcript_path)
            predicted_retention = self.pipeline.predict([original_intro])[0]
            self.log(f"Original Introduction: {original_intro}")
            self.log(f"Predicted Retention: {predicted_retention:.2f}%")

            # Generate multiple optimized introductions
            candidates = [
                self.generate_optimized_intro_gpt(original_intro)
            ]

            # Evaluate each candidate
            best_intro, best_retention = original_intro, predicted_retention
            for candidate in candidates:
                retention = self.pipeline.predict([candidate])[0]
                self.log(f"Candidate Introduction: {candidate}")
                self.log(f"Predicted Retention: {retention:.2f}%")
                if retention > best_retention:
                    best_intro, best_retention = candidate, retention

            self.display_best_intro(best_intro, best_retention)
        except Exception as e:
            self.log(f"Error during optimization: {e}")
        finally:
            loading_label.destroy()

    def display_best_intro(self, best_intro, best_retention):
        """
        Display the best introduction prominently in the output box.
        """
        best_intro_label = tk.Label(self.output_frame, text=f"Best Optimized Introduction: {best_intro}\nPredicted Retention: {best_retention:.2f}%",
                                    font=("Helvetica", 14, "bold"), fg="green", wraplength=1000, justify="left")
        best_intro_label.pack(pady=10)

    def generate_optimized_intro_gpt(self, original_intro):
        """
        Generate an optimized introduction using OpenAI's GPT model.
        """
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_KEY")

        prompt = [{
            'role': 'system',
            'content': f"Rewrite the following introduction to make it more engaging and improve audience retention: \n"
                       f"'{original_intro}'"
        }]

        response = openai.chat.completions.create(
            model='gpt-4o',
            messages=prompt,
        )
        optimized_intro = response.choices[0].message.content
        return optimized_intro

if __name__ == "__main__":
    root = tk.Tk()
    app = RetentionIntroOptimizationApp(root)
    root.mainloop()
