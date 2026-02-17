"""Professional Eye Blink Annotation Tool.

This script processes videos to detect eye blinks and exports their timestamps
and EAR data to TXT and NPY formats. It supports both single-video processing
and batch processing of directory trees.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.tree import Tree

from FaceMeshModule import FaceMeshGenerator
from utils import DrawingUtils, find_files

# Initialize Cyclopts and Rich
app = App(help="Professional Blink Annotation Tool")
console = Console()


@dataclass(frozen=True)
class PlannedOutputs:
    """Represents all output paths for a single input video."""

    video_path: Path
    output_dir: Path
    txt_path: Path
    npy_path: Path
    viz_path: Path | None
    post_txt_path: Path | None
    post_npy_path: Path | None


@dataclass(frozen=True)
class TreeEntry:
    """Represents a renderable leaf in a tree visualization."""

    path: Path
    label: str
    priority: int = 1


def _build_tree_visualization(entries: list[TreeEntry], root_dir: Path) -> Tree:
    """Build a rich tree visualization from entries anchored at root_dir."""
    tree = Tree(f":open_file_folder: [bold blue]{root_dir}[/bold blue]")
    dir_nodes = {root_dir: tree}

    sorted_entries = sorted(entries, key=lambda e: (str(e.path.parent), e.priority, e.path.name))

    for entry in sorted_entries:
        try:
            rel_path = entry.path.relative_to(root_dir)
        except ValueError:
            # Fallback for paths not relative to root
            tree.add(entry.label)
            continue

        current_node = tree
        current_path = root_dir

        # Iterate over parts except the last one (filename)
        for part in rel_path.parts[:-1]:
            current_path = current_path / part
            if current_path not in dir_nodes:
                dir_nodes[current_path] = current_node.add(
                    f":open_file_folder: [bold blue]{part}[/bold blue]"
                )
            current_node = dir_nodes[current_path]

        current_node.add(entry.label)

    return tree


def _resolve_output_directory(
    video_file: Path,
    *,
    input_root: Path | None,
    output_root: Path | None,
) -> Path:
    """Resolve the directory where outputs should be written for one input video."""
    if output_root is None:
        return video_file.parent

    if input_root is None:
        return output_root

    try:
        rel_parent = video_file.relative_to(input_root).parent
        return output_root / rel_parent
    except ValueError:
        logger.warning(
            "Could not map {} relative to {}. Falling back to {}",
            video_file,
            input_root,
            output_root,
        )
        return output_root


def _plan_outputs_for_video(
    video_file: Path,
    *,
    input_root: Path | None,
    output_root: Path | None,
    suffix: str,
    save_video: bool,
    postprocess: bool,
) -> PlannedOutputs:
    """Compute all output paths for one video."""
    current_out_dir = _resolve_output_directory(
        video_file, input_root=input_root, output_root=output_root
    )
    base_name = video_file.stem

    txt_path = current_out_dir / f"{base_name}{suffix}.txt"
    npy_path = current_out_dir / f"{base_name}{suffix}.npy"
    viz_path = current_out_dir / f"{base_name}{suffix}_viz.mp4" if save_video else None
    post_txt_path = (
        current_out_dir / f"{base_name}{suffix}_postprocessed.txt" if postprocess else None
    )
    post_npy_path = (
        current_out_dir / f"{base_name}{suffix}_postprocessed.npy" if postprocess else None
    )

    return PlannedOutputs(
        video_path=video_file,
        output_dir=current_out_dir,
        txt_path=txt_path,
        npy_path=npy_path,
        viz_path=viz_path,
        post_txt_path=post_txt_path,
        post_npy_path=post_npy_path,
    )


def _iter_output_artifacts(plan: PlannedOutputs) -> list[tuple[str, Path]]:
    """Return ordered artifact descriptors for display."""
    artifacts = [
        (":memo:", plan.txt_path),
        (":brain:", plan.npy_path),
    ]
    if plan.viz_path is not None:
        artifacts.append((":film_projector:", plan.viz_path))
    if plan.post_txt_path is not None:
        artifacts.append((":memo:", plan.post_txt_path))
    if plan.post_npy_path is not None:
        artifacts.append((":brain:", plan.post_npy_path))
    return artifacts


def _artifact_display(path: Path, *, force: bool) -> tuple[str, str]:
    """Return a status label and style for an output artifact path."""
    if force:
        return "overwrite", "bold bright_cyan"
    if path.exists():
        return "exists/skip", "bold yellow"
    return "create", "bold bright_cyan"


class BlinkAnnotator:
    """Handles the logic for processing video frames and detecting blink data."""

    def __init__(
        self,
        ear_threshold: float = 0.25,
        consec_frames: int = 3,
    ) -> None:
        """Initialize the annotator with detection parameters.

        Args:
            ear_threshold (float): The EAR value below which an eye is considered closed.
            consec_frames (int): Minimum consecutive frames to trigger a blink event.
        """
        self.generator = FaceMeshGenerator()
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames

        # Landmark indices for EAR calculation
        self.RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
        self.LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]

        # Plotting variables
        self.ear_history: list[float] = []
        self.frame_indices: list[int] = []
        self.max_plot_points = 100
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.canvas: FigureCanvas | None = None
        self.ear_line: Line2D | None = None

    def _init_plot(self, width: int, height: int = 400, dpi: int = 100) -> None:
        """Initialize the matplotlib figure with dimensions matching the video width.

        Args:
            width (int): Target width in pixels (matching video).
            height (int): Target height in pixels.
            dpi (int): Dots per inch for the figure.
        """
        plt.style.use("dark_background")
        figsize = (width / dpi, height / dpi)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)

        # Increase font sizes for better visibility in high-resolution exports
        self.ax.set_title("Eye Aspect Ratio (EAR) Over Time", fontsize=26, pad=20)
        self.ax.set_xlabel("Frame", fontsize=18)
        self.ax.set_ylabel("EAR", fontsize=18)

        # Set a fixed, wide range for the Y-axis
        self.ax.set_ylim(0.1, 0.5)

        (self.ear_line,) = self.ax.plot([], [], color="#56f10d", label="EAR", linewidth=3)
        self.ax.axhline(
            y=self.ear_threshold,
            color="#f70202",
            linestyle="--",
            label="Threshold",
            alpha=0.7,
            linewidth=2,
        )

        # Adjust tick label size
        self.ax.tick_params(axis="both", which="major", labelsize=18)
        self.ax.legend(loc="upper right", fontsize=18)
        self.ax.grid(True, alpha=0.2)
        self.fig.tight_layout()

    def _get_plot_image(self, current_frame: int, current_ear: float) -> np.ndarray:
        """Update the plot with new data and return it as a BGR image.

        Args:
            current_frame (int): The current frame index.
            current_ear (float): The calculated EAR for the current frame.

        Returns:
            np.ndarray: The plot as a BGR image.
        """
        if self.ax is None or self.canvas is None or self.ear_line is None:
            raise RuntimeError("Plot not initialized. Call _init_plot first.")

        self.ear_history.append(current_ear)
        self.frame_indices.append(current_frame)

        if len(self.ear_history) > self.max_plot_points:
            self.ear_history.pop(0)
            self.frame_indices.pop(0)

        self.ear_line.set_data(self.frame_indices, self.ear_history)
        self.ax.set_xlim(self.frame_indices[0], self.frame_indices[-1] + 1)

        self.canvas.draw()
        rgba_buffer = self.canvas.buffer_rgba()
        plot_img = np.asarray(rgba_buffer)

        return cv.cvtColor(plot_img, cv.COLOR_RGBA2BGR)

    def calculate_ear(
        self, eye_landmarks: list[int], landmarks: dict[int, tuple[int, int]]
    ) -> float:
        """Calculate the Eye Aspect Ratio (EAR).

        Args:
            eye_landmarks (list[int]): List of landmark indices for one eye.
            landmarks (dict): Dictionary of all detected face landmarks.

        Returns:
            The calculated EAR value.
        """
        p2_p6 = np.linalg.norm(
            np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]])
        )
        p3_p5 = np.linalg.norm(
            np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]])
        )
        p1_p4 = np.linalg.norm(
            np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]])
        )

        result = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return float(result)

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        save_video: bool = False,
        output_video_path: Optional[str] = None,
    ) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
        """Process a video file to detect blinks and collect EAR data.

        Args:
            video_path (str): Path to the input video file.
            start_time (float): Seconds from which to start processing.
            end_time (Optional[float]): Seconds at which to stop processing.
            save_video (bool): Whether to export a visualization video.
            output_video_path (Optional[str]): Destination path for the visualization video.

        Returns:
            tuple: A tuple containing:
                - detected_blinks: List of (timestamp, ear) for detected blinks (midpoint).
                - blink_ranges: List of lists, where each inner list contains (timestamp, ear)
                for all frames within a detected blink event.
        """
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")

        fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames
        num_frames_to_process = end_frame - start_frame

        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        # Setup Video Writer and Plot
        writer = None
        plot_height = 400
        if save_video and output_video_path:
            self._init_plot(width=width, height=plot_height)
            fourcc = cv.VideoWriter.fourcc(*"mp4v")
            writer = cv.VideoWriter(output_video_path, fourcc, fps, (width, height + plot_height))

        # Output Data Structures
        detected_blinks: list[tuple[float, float]] = []  # (timestamp, ear)
        blink_ranges: list[list[tuple[float, float]]] = []  # List of List of (timestamp, ear)
        current_blink_range: list[tuple[float, float]] = []

        blink_count = 0
        frame_counter = 0
        current_frame_idx = start_frame

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Processing {Path(video_path).name}...", total=num_frames_to_process
            )

            while cap.isOpened() and current_frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                _, face_landmarks = self.generator.create_face_mesh(frame, draw=False)
                avg_ear = 0.0
                timestamp = current_frame_idx / fps

                if face_landmarks:
                    r_ear = self.calculate_ear(self.RIGHT_EYE_EAR, face_landmarks)
                    l_ear = self.calculate_ear(self.LEFT_EYE_EAR, face_landmarks)
                    avg_ear = (r_ear + l_ear) / 2.0

                    if avg_ear < self.ear_threshold:
                        frame_counter += 1
                        current_blink_range.append((timestamp, avg_ear))
                    else:
                        if frame_counter >= self.consec_frames:
                            # Blink detected!
                            # Calculate midpoint for the main timestamp
                            mid_idx = len(current_blink_range) // 2
                            blink_midpoint = current_blink_range[mid_idx]

                            detected_blinks.append(
                                (round(blink_midpoint[0], 3), round(blink_midpoint[1], 3))
                            )
                            blink_ranges.append(current_blink_range)
                            blink_count += 1

                        # Reset
                        frame_counter = 0
                        current_blink_range = []

                if writer:
                    # Drawing
                    color = (30, 46, 209) if avg_ear < self.ear_threshold else (86, 241, 13)
                    if face_landmarks:
                        for idx in self.RIGHT_EYE_EAR + self.LEFT_EYE_EAR:
                            cv.circle(frame, face_landmarks[idx], 2, color, cv.FILLED)

                    # Update header text
                    header_text = (
                        f"Blinks: {blink_count} | EAR: {avg_ear:.3f} | "
                        f"t={timestamp:.3f} ({current_frame_idx})"
                    )
                    DrawingUtils.draw_text_with_bg(
                        frame,
                        header_text,
                        (10, 50),
                        font_scale=1.2,
                        thickness=3,
                        bg_color=color,
                    )

                    # Plotting
                    plot_img = self._get_plot_image(current_frame_idx, avg_ear)
                    combined = cv.vconcat([frame, plot_img])
                    writer.write(combined)

                current_frame_idx += 1
                progress.update(task, advance=1)

        cap.release()
        if writer:
            writer.release()

        return detected_blinks, blink_ranges

    @staticmethod
    def postprocess_blinks(
        blink_ranges: list[list[tuple[float, float]]],
    ) -> list[tuple[float, float]]:
        """Find the frame with the lowest EAR for each blink range.

        Args:
            blink_ranges: List of blink events, where each event is a list of (timestamp, ear).

        Returns:
            List of (timestamp, ear) tuples corresponding to the lowest EAR in each blink.
        """
        postprocessed = []
        for blink_range in blink_ranges:
            if not blink_range:
                continue
            # Find the tuple with the minimum EAR (second element)
            min_ear_Frame = min(blink_range, key=lambda x: x[1])
            postprocessed.append((round(min_ear_Frame[0], 3), round(min_ear_Frame[1], 3)))
        return postprocessed


@app.default
def main(
    video_path: Optional[str] = None,
    *,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    ear_threshold: float = 0.22,
    consec_frames: int = 2,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    quick_test: bool = False,
    save_video: bool = False,
    postprocess: bool = False,
    depth: Optional[int] = None,
    pattern: Optional[str] = None,
    suffix: str = "_eyeblink",
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Annotate videos for eye blinks and save timestamps + EAR data.

    Can process a single video or recursively search a directory.

    Args:
        video_path: Path to a single input video file.
        input_dir: Base directory to search for videos (mutually exclusive with video_path).
        output_dir: Directory for results. If None, saves alongside source video.
        ear_threshold: EAR sensitivity (lower is less sensitive).
        consec_frames: Minimum frames to count as a blink.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        quick_test: Process only first 5 seconds.
        save_video: Export visualization video.
        postprocess: Find lowest EAR frame per blink range.
        depth: Recursion depth for input_dir (0=root only).
        pattern: Glob pattern for filtering filenames (e.g. "*subject-01*").
        suffix: Suffix for output files.
        force: Overwrite existing output files.
        dry_run: If True, list files to be processed and exit.
    """
    if video_path is None and input_dir is None:
        console.print("[bold red]Error:[/bold red] Must provide either VIDEO_PATH or --input-dir.")
        return

    # Collect videos to process
    videos_to_process: list[Path] = []
    if input_dir:
        input_path = Path(input_dir)
        console.print(f"[bold]Searching for videos in {input_path}...[/bold]")
        found_videos: list[Path] = find_files(
            folder=input_path,
            extensions=["mp4", "avi", "mov", "mkv"],
            name_patterns=pattern,
            depth=depth,
            as_path=True,
        )
        videos_to_process.extend(found_videos)
        if not videos_to_process:
            logger.warning(f"No videos found in {input_path} matching criteria.")
            return
    elif video_path:
        # Single video mode
        videos_to_process.append(Path(video_path))

    # Resolve and plan outputs once so dry-run and processing share the same logic.
    videos_to_process = sorted(v.resolve() for v in videos_to_process)
    input_root: Path = (
        Path(input_dir).resolve() if input_dir else videos_to_process[0].parent.resolve()
    )
    output_root: Path | None = Path(output_dir).resolve() if output_dir else None
    plans: list[PlannedOutputs] = [
        _plan_outputs_for_video(
            video_file,
            input_root=input_root if input_dir else None,
            output_root=output_root,
            suffix=suffix,
            save_video=save_video,
            postprocess=postprocess,
        )
        for video_file in videos_to_process
    ]
    plans_by_video: dict[Path, PlannedOutputs] = {plan.video_path: plan for plan in plans}

    # Visualize trees
    if plans:
        try:
            output_is_separate: bool = output_root is not None and output_root != input_root

            video_entries: list[TreeEntry] = [
                TreeEntry(
                    path=plan.video_path,
                    label=f":movie_camera: [bold green]{plan.video_path.name}[/bold green]",
                    priority=0,
                )
                for plan in plans
            ]

            if dry_run and not output_is_separate:
                combined_entries: list[TreeEntry] = list(video_entries)
                for plan in plans:
                    for icon, artifact_path in _iter_output_artifacts(plan):
                        status, style = _artifact_display(artifact_path, force=force)
                        combined_entries.append(
                            TreeEntry(
                                path=artifact_path,
                                label=(
                                    f"{icon} [{style}]{artifact_path.name}[/{style}] "
                                    f"[dim]({status})[/dim]"
                                ),
                                priority=1,
                            )
                        )
                console.print(_build_tree_visualization(combined_entries, input_root))
            else:
                console.print(_build_tree_visualization(video_entries, input_root))
                if dry_run and output_is_separate and output_root is not None:
                    console.print("\n[bold]Output tree (planned artifacts)[/bold]")
                    output_entries: list[TreeEntry] = []
                    for plan in plans:
                        for icon, artifact_path in _iter_output_artifacts(plan):
                            status, style = _artifact_display(artifact_path, force=force)
                            output_entries.append(
                                TreeEntry(
                                    path=artifact_path,
                                    label=(
                                        f"{icon} [{style}]{artifact_path.name}[/{style}] "
                                        f"[dim]({status})[/dim]"
                                    ),
                                )
                            )
                    console.print(_build_tree_visualization(output_entries, output_root))
        except Exception:
            logger.warning("Could not build directory tree visualization.")

    console.print(f"\n[green]Found {len(videos_to_process)} videos to process.[/green]\n")

    if dry_run:
        console.print("[yellow]Dry run enabled. Exiting without processing.[/yellow]")
        return

    if quick_test:
        console.print("[yellow]Quick test mode enabled. Processing 5 seconds per video.[/yellow]")
        end_time = start_time + 5.0

    annotator = BlinkAnnotator(ear_threshold=ear_threshold, consec_frames=consec_frames)

    for video_file in videos_to_process:
        try:
            plan = plans_by_video[video_file]

            # Check existence
            if not force and plan.txt_path.exists() and plan.npy_path.exists():
                logger.warning(
                    f"Skipping {video_file.name} - output exists (use --force to overwrite)"
                )
                continue

            plan.output_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"Processing: [blue]{video_file}[/blue]")

            # PROCESS
            timestamps, blink_ranges = annotator.process(
                str(video_file),
                start_time=start_time,
                end_time=end_time,
                save_video=save_video,
                output_video_path=str(plan.viz_path) if plan.viz_path else None,
            )

            # Save basic results
            # TXT: timestamp <TAB> ear
            with plan.txt_path.open("w") as f:
                for ts, ear in timestamps:
                    f.write(f"{ts}\t{ear}\n")

            # NPY: shape (N, 2)
            if timestamps:
                np.save(plan.npy_path, np.array(timestamps))
            else:
                np.save(plan.npy_path, np.empty((0, 2)))

            # Handle Post-processing
            if postprocess:
                pp_timestamps = annotator.postprocess_blinks(blink_ranges)

                if plan.post_txt_path is None or plan.post_npy_path is None:
                    raise RuntimeError("Postprocess paths were not planned correctly.")

                with plan.post_txt_path.open("w") as f:
                    for ts, ear in pp_timestamps:
                        f.write(f"{ts}\t{ear}\n")

                if pp_timestamps:
                    np.save(plan.post_npy_path, np.array(pp_timestamps))
                else:
                    np.save(plan.post_npy_path, np.empty((0, 2)))

            console.print(
                f"[bold green]Done![/bold green] {len(timestamps)} blinks. "
                f"Saved to {plan.output_dir}"
            )

        except Exception as e:
            logger.exception(f"Failed to process {video_file}: {e}")
            continue


if __name__ == "__main__":
    app()
