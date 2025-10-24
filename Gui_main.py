import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safer backend for macOS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as tb  # pip install ttkbootstrap

# --- import your functions ---
from main import loop, CostMatrix, tsp_annealing


class BPMGUI(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Playlist Optimizer")
        self.geometry("900x750")

        # === PATH INPUT ===
        path_frame = ttk.Frame(self)
        path_frame.pack(pady=10, fill="x", padx=20)

        ttk.Label(path_frame, text="Data Path:").pack(side="left", padx=5)
        self.path_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.path_var, width=70).pack(side="left", padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_path).pack(side="left")

        # === RUN BUTTON ===
        self.run_btn = ttk.Button(self, text="Run Analysis", command=self.run_analysis_thread, bootstyle="success")
        self.run_btn.pack(pady=10)

        # === PROGRESS BAR ===
        prog_frame = ttk.Frame(self)
        prog_frame.pack(fill="x", padx=20)
        self.progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.progress_label = ttk.Label(prog_frame, text="Idle")
        self.progress_label.pack(side="right")

        # === LOG OUTPUT ===
        self.output_box = tk.Text(self, height=5, bg="#1c1c1c", fg="white", insertbackground="white")
        self.output_box.pack(padx=20, pady=10, fill="x", expand=False)

        # === TABLE FRAME ===
        table_frame = ttk.Frame(self)
        table_frame.pack(padx=20, pady=10, fill="both", expand=False)

        columns = ("#", "name", "bpm", "key")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        self.tree.heading("#", text="#")
        self.tree.heading("name", text="Song Name")
        self.tree.heading("bpm", text="BPM")
        self.tree.heading("key", text="Key")

        for col in columns:
            self.tree.column(col, anchor="center", width=160 if col == "Name" else 80)

        style = ttk.Style()
        style.configure("Treeview",
                        background="#1e1e1e",
                        fieldbackground="#1e1e1e",
                        foreground="white",
                        rowheight=25)
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))
        style.map("Treeview", background=[("selected", "#007acc")])

        self.tree.pack(fill="both", expand=True)

        # === MATPLOTLIB PLOT AREA ===
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill="both", expand=True)

        self.df = None

    # -----------------------------------------------------------------

    def browse_path(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_var.set(folder)

    def log(self, msg):
        self.output_box.insert(tk.END, msg + "\n")
        self.output_box.see(tk.END)
        self.update_idletasks()

    # -----------------------------------------------------------------
    # Thread-safe workflow

    def run_analysis_thread(self):
        thread = threading.Thread(target=self._background_task)
        thread.start()

    def _background_task(self):
        path = self.path_var.get().strip()
        if not path or not Path(path).exists():
            self.after(0, lambda: messagebox.showerror("Error", "Please select a valid directory."))
            return

        self.after(0, lambda: self.prepare_ui_for_run(path))

        try:
            p_name, bpms, keys = self.loop_with_progress(path)
            df = pd.DataFrame({"name": p_name, "bpm": bpms, "key": keys})
            cost_matrix = CostMatrix(df)
            cost_matrix.compute_matrix()

            if len(df) < 3:
                self.after(0, lambda: self.log("Not enough songs to solve TSP."))
                return

            best_cost = float("inf")
            best_perm = None
            for _ in range(10):
                perm, cost = tsp_annealing(cost_matrix.matrix)
                if cost < best_cost:
                    best_perm, best_cost = perm, cost

            ordered_df = df.iloc[best_perm].reset_index(drop=True)
            self.after(0, lambda: self.update_gui_with_results(ordered_df))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: self.cleanup_after_run())

    # -----------------------------------------------------------------
    # GUI updates

    def prepare_ui_for_run(self, path):
        self.run_btn.config(state="disabled")
        self.output_box.delete("1.0", tk.END)
        self.log(f"Extracting features from {path}...")
        self.progress["value"] = 0
        self.progress_label.config(text="Starting...")
        for row in self.tree.get_children():
            self.tree.delete(row)

    def cleanup_after_run(self):
        self.progress_label.config(text="Done")
        self.run_btn.config(state="normal")

    def update_progress(self, i, total):
        percent = int((i / total) * 100)
        self.progress["value"] = percent
        self.progress_label.config(text=f"{percent}%")
        self.update_idletasks()

    def update_gui_with_results(self, ordered_df):
        self.df = ordered_df
        self.log("✅ Analysis complete!")

        # Populate table
        for i, row in ordered_df.iterrows():
            self.tree.insert("", "end", values=(i + 1, row["name"], f"{row['bpm']:.2f}", row["key"]))

        # Clear previous plot
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig = self.plot_bpm_key_variation_embed(ordered_df)
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # -----------------------------------------------------------------
    # Modified loop with progress

    def loop_with_progress(self, path: str):
        from main import sound_C, BPM_extractor, predict_key

        p_name, bpms, keys = [], [], []
        files = list(Path(path).glob("*.[mw][ap][3v]"))
        total = len(files)

        for i, file in enumerate(files, start=1):
            sound = sound_C(file)
            sr, audio = sound.fs, sound.sound
            Extractor = BPM_extractor(sr, novelty="spectral", tempogram="fourier",
                                      tempo_w=30, tempo_hop=3, novelty_w=5.0, novelty_hop=0.05)
            bpm = Extractor.get_BPM(audio, plot=False)
            bpms.append(bpm)

            pred = predict_key(file)
            keys.append(pred)

            p_name.append(file.stem)
            self.after(0, lambda i=i: self.update_progress(i, total))

        return p_name, bpms, keys

    def plot_bpm_key_variation_embed(self, df, bpm_low=100.0, bpm_high=190.0):
        """
        - Folds BPM by powers of two into [bpm_low, bpm_high).
        - Maps Camelot keys 1A–12B to numbers 1–12 (ignoring A/B).
        - Plots BPM (folded) and key wheel position on same sequence axis.
        - Dark theme with white text.
        """
        import numpy as np

        def fold_bpm(bpm, low=bpm_low, high=bpm_high):
            """Fold bpm by doubling/halving (powers of 2) into [low, high)."""
            if np.isnan(bpm):
                return bpm
            while bpm >= high:
                bpm /= 2.0
            while bpm < low:
                bpm *= 2.0
            return bpm

        def camelot_to_num(k):
            """Map 1A–12B → 1–12 (ignore A/B side)."""
            if not isinstance(k, str):
                return np.nan
            try:
                num = int(''.join(ch for ch in k if ch.isdigit()))
                return num
            except ValueError:
                return np.nan

        # --- prep data ---
        df = df.copy()
        df["Seq"] = np.arange(1, len(df) + 1)
        df["BPM_folded"] = df["bpm"].apply(fold_bpm)
        df["Key_num"] = df["key"].apply(camelot_to_num)

        # --- plot ---
        fig, ax1 = plt.subplots(figsize=(10, 4), facecolor="#222")
        ax1.set_facecolor("#222")

        # BPM (folded)
        ax1.plot(df["Seq"], df["BPM_folded"], "o-", color="tab:blue", label=f"BPM")
        ax1.set_ylabel("BPM (folded)", color="white")
        ax1.set_xlabel("Song Sequence", color="white")
        ax1.set_xticks(df["Seq"])
        ax1.set_xticklabels(df["Seq"], color="white")
        ax1.tick_params(axis='y', colors="white")
        ax1.grid(alpha=0.3, color="gray")

        # Keys (1–12)
        ax2 = ax1.twinx()
        ax2.plot(df["Seq"], df["Key_num"], "s-", color="tab:red", label="Key (1–12)")
        ax2.set_ylabel("Key (Camelot wheel position)", color="white")
        ax2.tick_params(axis='y', colors="white")
        ax2.set_ylim(0.5, 12.5)

        ax2.set_yticks(range(1, 13))
        ax2.set_yticklabels([str(i) for i in range(1, 13)], color="white")

        # Styling
        for ax in (ax1, ax2):
            for spine in ax.spines.values():
                spine.set_color("white")

        ax1.legend(loc="upper left", facecolor="#222", edgecolor="white", labelcolor="white")
        ax2.legend(loc="upper right", facecolor="#222", edgecolor="white", labelcolor="white")
        plt.title("BPM (folded ×2) and Key Progression (1–12 Camelot Wheel)", color="white")
        fig.tight_layout()
        return fig

if __name__ == "__main__":
    app = BPMGUI()
    app.mainloop()