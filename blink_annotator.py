"""Professional Eye Blink Annotation Tool.

This script processes a video to detect eye blinks and exports their timestamps
to TXT and NPY formats. It uses MediaPipe for landmark detection and Cyclopts
for the CLI interface.
"""

from pathlib import Path

import cv2 as cv
import numpy as np
from cyclopts import App
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from FaceMeshModule import FaceMeshGenerator

# Initialize Cyclopts and Rich
app = App(help="Professional Blink Annotation Tool")
console = Console()


class BlinkAnnotator:
    """Handles the logic for processing video frames and detecting blink timestamps."""

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

    def calculate_ear(self, eye_landmarks: list[int], landmarks: dict) -> float:
        """Calculate the Eye Aspect Ratio (EAR).

        Args:
            eye_landmarks (list[int]): List of landmark indices for one eye.
            landmarks (dict): Dictionary of all detected face landmarks.

        Returns:
            The calculated EAR value.
        """
        # Vertical distances
        p2_p6 = np.linalg.norm(
            np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]])
        )
        p3_p5 = np.linalg.norm(
            np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]])
        )
        # Horizontal distance
        p1_p4 = np.linalg.norm(
            np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]])
        )

        result = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return float(result)

    def process(
        self, video_path: str, start_time: float = 0.0, end_time: float | None = None
    ) -> list[float]:
        """Process a video file and returns a list of blink timestamps.

        Args:
            video_path: Path to the input video file.
            start_time: Seconds from which to start processing.
            end_time: Seconds at which to stop processing.

        Returns:
            A list of timestamps (seconds) where blinks were detected.
        """
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")

        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # Calculate frame boundaries
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames
        num_frames_to_process = end_frame - start_frame

        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        blink_timestamps: list[float] = []
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
            task = progress.add_task("[cyan]Detecting blinks...", total=num_frames_to_process)

            while cap.isOpened() and current_frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                _, face_landmarks = self.generator.create_face_mesh(frame, draw=False)

                if face_landmarks:
                    r_ear = self.calculate_ear(self.RIGHT_EYE_EAR, face_landmarks)
                    l_ear = self.calculate_ear(self.LEFT_EYE_EAR, face_landmarks)
                    avg_ear = (r_ear + l_ear) / 2.0

                    if avg_ear < self.ear_threshold:
                        frame_counter += 1
                    else:
                        if frame_counter >= self.consec_frames:
                            # Use the timestamp from the middle of the blink duration
                            timestamp = (current_frame_idx - (frame_counter // 2)) / fps
                            blink_timestamps.append(round(timestamp, 3))
                        frame_counter = 0

                current_frame_idx += 1
                progress.update(task, advance=1)

        cap.release()
        return blink_timestamps


@app.default
def main(
    video_path: str,
    output_dir: Path | str = "annotations",
    ear_threshold: float = 0.22,
    consec_frames: int = 2,
    start_time: float = 0.0,
    end_time: float | None = None,
    *,
    quick_test: bool = False,
) -> None:
    """Annotate a video for eye blinks and saves timestamps to disk.

    Args:
        video_path (Path | str): Path to the input .mp4 or .avi file.
        output_dir (Path | str): Directory where results will be saved. Default is "annotations".
        ear_threshold (float): Sensitivity for blink detection (lower is less sensitive).
            Default is 0.22, which is a common threshold for EAR-based blink detection.
        consec_frames (int): How many frames the eye must be closed to count as a blink.
            Default is 2.
        start_time (float): Time in seconds to start detection.
            Default is 0.0 (beginning of the video).
        end_time (float | None): Time in seconds to end detection.
            Default is None (end of the video).
        quick_test (bool): If True, only processes the first 5 seconds of the video.
            Default is False.
    """
    if quick_test:
        console.print("[yellow]Quick test mode enabled. Processing 5 seconds only.[/yellow]")
        end_time = start_time + 5.0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(video_path).stem

    annotator = BlinkAnnotator(ear_threshold=ear_threshold, consec_frames=consec_frames)

    try:
        timestamps = annotator.process(video_path, start_time, end_time)

        # Save TXT
        txt_path = output_dir / f"{base_name}_blinks.txt"
        with txt_path.open("w") as f:
            f.write("\n".join(map(str, timestamps)))

        # Save NPY
        npy_path = output_dir / f"{base_name}_blinks.npy"
        np.save(npy_path, np.array(timestamps))

        console.print(
            f"\n[bold green]Success![/bold green] Detected [bold]{len(timestamps)}[/bold] blinks."
        )
        console.print(f"Results saved to: [blue]{output_dir}[/blue]")

    except Exception as e:
        logger.error(f"Processing failed: {e}")


if __name__ == "__main__":
    app()
