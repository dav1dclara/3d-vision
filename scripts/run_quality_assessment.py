"""
Interactive GUI for Mesh Quality Assessment

This script provides a minimal graphical interface to:
1. Select a mesh file (.ply)
2. Select a point cloud file (.ply)
3. Run quality assessment via evaluate_mesh()
4. Display results and visualization

Usage:
    python scripts/run_quality_assessment.py
"""

import gc
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from contextlib import redirect_stdout, redirect_stderr
from queue import Queue, Empty
import threading
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quality_assessment import (
    _load_ply_mesh,
    _load_ply_pointcloud,
    _count_ply_points,
    evaluate_mesh,
)


class QualityAssessmentGUI:
    """Minimal GUI for mesh quality assessment workflow."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Mesh Quality Assessment")
        self.root.geometry("900x820")
        self.root.minsize(900, 820)
        
        self.mesh_path = None
        self.pointcloud_path = None
        self.point_count = None
        self.last_plot = None
        self.log_queue = Queue()
        self.metric_checks = []
        
        # ── UI Layout ──────────────────────────────────────────────────
        
        # Title
        title = tk.Label(root, text="Mesh Quality Assessment", font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        # Mesh selection
        mesh_frame = tk.Frame(root)
        mesh_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(mesh_frame, text="Mesh (.ply):", font=("Arial", 10)).pack(side="left")
        self.mesh_label = tk.Label(
            mesh_frame, text="No file selected", fg="gray", font=("Arial", 9)
        )
        self.mesh_label.pack(side="left", padx=10, fill="x", expand=True)
        
        tk.Button(
            mesh_frame, text="Browse", command=self.select_mesh, width=12
        ).pack(side="right")
        
        # Point cloud selection
        pc_frame = tk.Frame(root)
        pc_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(pc_frame, text="Point Cloud (.ply):", font=("Arial", 10)).pack(side="left")
        self.pc_label = tk.Label(
            pc_frame, text="No file selected", fg="gray", font=("Arial", 9)
        )
        self.pc_label.pack(side="left", padx=10, fill="x", expand=True)
        
        tk.Button(
            pc_frame, text="Browse", command=self.select_pointcloud, width=12
        ).pack(side="right")
        
        # Separator
        tk.Frame(root, height=2, bd=1, relief="sunken").pack(fill="x", pady=20, padx=20)
        
        # Options frame
        options_frame = tk.LabelFrame(root, text="Options", font=("Arial", 10, "bold"), padx=10, pady=10)
        options_frame.pack(pady=10, padx=20, fill="x")
        
        # Sample size
        sample_frame = tk.Frame(options_frame)
        sample_frame.pack(fill="x", pady=5)
        tk.Label(sample_frame, text="Max sample size:", width=15).pack(side="left")
        self.sample_var = tk.StringVar(value="50000")
        self.sample_entry = tk.Entry(sample_frame, textvariable=self.sample_var, width=10)
        self.sample_entry.pack(side="left", padx=5)
        self.use_all_points_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            sample_frame,
            text="Use all points (no subsampling)",
            variable=self.use_all_points_var,
            command=self._toggle_sample_entry,
        ).pack(side="left", padx=10)
        
        # Thresholds
        thresh_frame = tk.Frame(options_frame)
        thresh_frame.pack(fill="x", pady=5)
        tk.Label(thresh_frame, text="Thresholds (cm):", width=15).pack(side="left")
        self.thresh1_var = tk.StringVar(value="1.0")
        self.thresh2_var = tk.StringVar(value="2.0")
        self.thresh3_var = tk.StringVar(value="10.0")
        thresh_entries = tk.Frame(thresh_frame)
        thresh_entries.pack(side="left", padx=(5, 0))

        zero_col = tk.Frame(thresh_entries)
        zero_col.pack(side="left")
        tk.Label(zero_col, text="0", fg="gray").pack()
        tk.Label(zero_col, text="", fg="gray").pack()

        good_col = tk.Frame(thresh_entries)
        good_col.pack(side="left", padx=12)
        tk.Label(good_col, text="-", fg="gray").pack()
        tk.Label(good_col, text="Good", fg="gray").pack()

        t1_col = tk.Frame(thresh_entries)
        t1_col.pack(side="left")
        tk.Entry(t1_col, textvariable=self.thresh1_var, width=6).pack()

        ok_col = tk.Frame(thresh_entries)
        ok_col.pack(side="left", padx=12)
        tk.Label(ok_col, text="-", fg="gray").pack()
        tk.Label(ok_col, text="OK", fg="gray").pack()

        t2_col = tk.Frame(thresh_entries)
        t2_col.pack(side="left")
        tk.Entry(t2_col, textvariable=self.thresh2_var, width=6).pack()

        critical_col = tk.Frame(thresh_entries)
        critical_col.pack(side="left", padx=12)
        tk.Label(critical_col, text="-", fg="gray").pack()
        tk.Label(critical_col, text="Critical", fg="gray").pack()

        t3_col = tk.Frame(thresh_entries)
        t3_col.pack(side="left")
        tk.Entry(t3_col, textvariable=self.thresh3_var, width=6).pack()

        # Metrics selection
        metrics_frame = tk.LabelFrame(options_frame, text="Metrics", padx=10, pady=5)
        metrics_frame.pack(fill="x", pady=10)

        self.structure_var = tk.BooleanVar(value=True)
        self.distance_var = tk.BooleanVar(value=True)
        self.residual_var = tk.BooleanVar(value=True)
        self.watertight_var = tk.BooleanVar(value=False)
        self.fscore_var = tk.BooleanVar(value=False)
        self.visual_var = tk.BooleanVar(value=True)

        checks = [
            ("Structure", self.structure_var),
            ("Distance", self.distance_var),
            ("Residual Distribution", self.residual_var),
            ("Watertight/manifold", self.watertight_var),
            ("F-Score", self.fscore_var),
            ("Visualization", self.visual_var),
        ]

        for idx, (label, var) in enumerate(checks):
            chk = tk.Checkbutton(metrics_frame, text=label, variable=var)
            chk.grid(row=idx // 2, column=idx % 2, sticky="w", padx=5, pady=2)
            self.metric_checks.append(chk)
        
        # Separator
        tk.Frame(root, height=1, bd=1, relief="sunken").pack(fill="x", pady=10, padx=20)

        # Status
        status_frame = tk.Frame(root)
        status_frame.pack(fill="x", padx=20)
        tk.Label(status_frame, text="Status:", width=8, anchor="w").pack(side="left")
        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)

        # Log output
        self.log_box = scrolledtext.ScrolledText(root, height=14, wrap="word", state="disabled")
        self.log_box.pack(fill="both", padx=20, pady=10, expand=True)

        # Action buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame, text="Evaluate", command=self.run_evaluation,
            bg="green", fg="white", font=("Arial", 11, "bold"), width=15
        ).pack(side="left", padx=10)
        
        self.plot_button = tk.Button(
            button_frame, text="Open Plot", command=self.open_plot,
            width=15, state="disabled"
        )
        self.plot_button.pack(side="left", padx=10)

        self.clear_button = tk.Button(
            button_frame, text="Clear", command=self.clear_selection,
            width=15
        )
        self.clear_button.pack(side="left", padx=10)
        
        self.exit_button = tk.Button(
            button_frame, text="Exit", command=root.quit,
            bg="red", fg="white", width=15
        )
        self.exit_button.pack(side="left", padx=10)

        self._poll_log_queue()

    def _set_status(self, text):
        """Update status line."""
        self.status_var.set(text)
        self.root.update_idletasks()

    def _log(self, text):
        """Append text to the log output."""
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _toggle_sample_entry(self):
        """Enable/disable sample size entry based on checkbox."""
        if self.use_all_points_var.get():
            if self.point_count is not None:
                self.sample_entry.config(state="normal")
                self.sample_var.set(str(self.point_count))
                self.sample_entry.config(state="disabled")
                self.root.update_idletasks()
            else:
                self.sample_entry.config(state="disabled")
        else:
            self.sample_entry.config(state="normal")
            self.sample_var.set("50000")

    def _poll_log_queue(self):
        """Flush queued log messages into the GUI."""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._log(msg)
        except Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _enqueue_log(self, text):
        """Thread-safe log enqueue."""
        self.log_queue.put(text)
    
    def select_mesh(self):
        """Open file dialog to select mesh."""
        path = filedialog.askopenfilename(
            title="Select Mesh File",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            initialdir="outputs"
        )
        if path:
            self.mesh_path = path
            filename = Path(path).name
            self.mesh_label.config(text=filename, fg="black")
    
    def select_pointcloud(self):
        """Open file dialog to select point cloud."""
        path = filedialog.askopenfilename(
            title="Select Point Cloud File",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
            initialdir="outputs"
        )
        if path:
            self.pointcloud_path = path
            filename = Path(path).name
            self.pc_label.config(text=filename, fg="black")
            try:
                self.point_count = _count_ply_points(self.pointcloud_path)
                if self.use_all_points_var.get():
                    self.sample_entry.config(state="normal")
                    self.sample_var.set(str(self.point_count))
                    self.sample_entry.config(state="disabled")
                    self.root.update_idletasks()
            except Exception:
                self.point_count = None
    
    def clear_selection(self):
        """Clear all selections."""
        self.mesh_path = None
        self.pointcloud_path = None
        self.last_plot = None
        self.mesh_label.config(text="No file selected", fg="gray")
        self.pc_label.config(text="No file selected", fg="gray")
        self.plot_button.config(state="disabled")
        self._set_status("Ready.")
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def open_plot(self):
        """Open the last plot in the browser."""
        if self.last_plot is None:
            messagebox.showinfo("Info", "No plot available yet.")
            return
        try:
            self.last_plot.show()
        except Exception as e:
            messagebox.showerror("Error", f"Could not display plot:\n{e}")
    
    def run_evaluation(self):
        """Run quality assessment."""
        # ── Validation ─────────────────────────────────────────────────
        if not self.mesh_path:
            messagebox.showerror("Error", "Please select a mesh file")
            return
        
        if not self.pointcloud_path:
            messagebox.showerror("Error", "Please select a point cloud file")
            return
        
        # ── Parse options ──────────────────────────────────────────────
        if self.use_all_points_var.get():
            sample_size = None
        else:
            try:
                sample_size = int(self.sample_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid sample size (must be integer)")
                return
        
        try:
            t1 = float(self.thresh1_var.get())
            t2 = float(self.thresh2_var.get())
            t3 = float(self.thresh3_var.get())
            thresholds_cm = [t1, t2, t3]
            if not (t1 < t2 < t3):
                raise ValueError("Thresholds must be increasing (t1 < t2 < t3)")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid thresholds: {e}")
            return
        
        # ── Load files ─────────────────────────────────────────────────
        try:
            self._set_status("Loading files...")
            self._log("LOG:")
            self._log("    Loading data...")
            mesh = _load_ply_mesh(self.mesh_path)
            # Pass max_points so only the needed sample is paged into RAM.
            # When "use all points" is active, sample_size is None → full load.
            points = _load_ply_pointcloud(self.pointcloud_path,
                                          max_points=sample_size)
            if self.point_count is None:
                self.point_count = len(points)
            if self.use_all_points_var.get():
                sample_size = self.point_count
                self.sample_entry.config(state="normal")
                self.sample_var.set(str(self.point_count))
                self.sample_entry.config(state="disabled")
                self.root.update_idletasks()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load files:\n{e}")
            self._log(f"ERROR: {e}")
            self._set_status("Failed to load files.")
            return

        # ── Run evaluation in background ───────────────────────────────
        self._set_status("Evaluating...")
        self._set_controls_state("disabled")

        thread = threading.Thread(
            target=self._evaluate_worker,
            args=(
                mesh,
                points,
                sample_size,
                thresholds_cm,
                self.structure_var.get(),
                self.distance_var.get(),
                self.residual_var.get(),
                self.watertight_var.get(),
                self.fscore_var.get(),
                self.visual_var.get(),
            ),
            daemon=True,
        )
        thread.start()
        del mesh, points  # release GUI-side references; thread holds the only copies now
        gc.collect()

    def _set_controls_state(self, state):
        """Enable/disable controls while running."""
        if state == "disabled":
            self.plot_button.config(state="disabled")
        else:
            self.plot_button.config(state="normal" if self.last_plot is not None else "disabled")
        self.clear_button.config(state=state)
        self.exit_button.config(state=state)
        for chk in self.metric_checks:
            chk.config(state=state)

    def _evaluate_worker(
        self,
        mesh,
        points,
        sample_size,
        thresholds_cm,
        compute_structure,
        compute_distance,
        compute_residual_distribution,
        compute_watertightness,
        compute_f_score,
        compute_visualization,
    ):
        """Background worker to run evaluation and stream logs."""
        try:
            writer = _QueueWriter(self._enqueue_log)
            with redirect_stdout(writer), redirect_stderr(writer):
                results = evaluate_mesh(
                    mesh=mesh,
                    ground_truth_points=points,
                    sample_size=sample_size,
                    thresholds_cm=thresholds_cm,
                    seed=42,
                    verbose=True,
                    detailed_summary=True,
                    mesh_label=Path(self.mesh_path).name,
                    pointcloud_label=Path(self.pointcloud_path).name,
                    compute_structure=compute_structure,
                    compute_distance=compute_distance,
                    compute_residual_distribution=compute_residual_distribution,
                    compute_watertightness=compute_watertightness,
                    compute_f_score=compute_f_score,
                    compute_visualization=compute_visualization,
                    original_point_count=self.point_count,
                )
            self._enqueue_log("")

            self.last_plot = results.get("mesh_with_colors")
            if self.last_plot is not None:
                self._enqueue_log("Plot is ready. Click 'Open Plot' to view.")

            del mesh, points
            gc.collect()

            self.root.after(0, self._evaluation_success, results)

        except Exception as e:
            del mesh, points
            gc.collect()
            self._enqueue_log(f"ERROR: {e}")
            self.root.after(0, self._evaluation_failed, e)

    def _evaluation_success(self, results):
        """Finalize UI after success."""
        if self.last_plot is not None:
            self.plot_button.config(state="normal")
        self._set_controls_state("normal")
        self._set_status("Done.")

    def _evaluation_failed(self, error):
        """Finalize UI after failure."""
        self._set_controls_state("normal")
        self._set_status("Evaluation failed.")
        messagebox.showerror("Error", f"Evaluation failed:\n{error}")


class _QueueWriter:
    """File-like writer that streams lines into a queue."""

    def __init__(self, enqueue_func):
        self.enqueue = enqueue_func
        self._buffer = ""

    def write(self, text):
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                if line.startswith("Results:"):
                    self.enqueue(line)
                else:
                    self.enqueue("    " + line)
            else:
                self.enqueue("")
        return len(text)

    def flush(self):
        if self._buffer:
            self.enqueue(self._buffer)
            self._buffer = ""


def main():
    """Launch the GUI."""
    root = tk.Tk()
    gui = QualityAssessmentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
