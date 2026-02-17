"""A module to generate face mesh visualizations using MediaPipe's modern Tasks API.

This is used by blink_annotator.py for landmark detection.
"""

import os
import urllib.request
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils


class FaceMeshGenerator:
    """
    A class to detect face landmarks using the modern MediaPipe Tasks API.
    """

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        num_faces: int = 2,
        min_detection_con: float = 0.5,
        min_track_con: float = 0.5,
        running_mode=vision.RunningMode.IMAGE,
    ) -> None:
        """
        Initialize the FaceLandmarker detector.

        Args:
            model_path: Path to the .task model file.
            num_faces: Maximum number of faces to detect.
            min_detection_con: Minimum confidence score for face detection.
            min_track_con: Minimum confidence score for face presence (tracking).
            running_mode: MediaPipe running mode (IMAGE, VIDEO, or LIVE_STREAM).
        """
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Downloading...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")

        try:
            self.running_mode = running_mode
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=self.running_mode,
                num_faces=num_faces,
                min_face_detection_confidence=min_detection_con,
                min_face_presence_confidence=min_track_con,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FaceLandmarker: {e}")

    def create_face_mesh(
        self, frame: np.ndarray, draw: bool = True, timestamp_ms: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
        """
        Process a frame to detect face landmarks.

        Args:
            frame: Input BGR image from OpenCV.
            draw: Whether to draw mesh contours on the frame.
            timestamp_ms: Required if running_mode is VIDEO.

        Returns:
            A tuple containing the processed frame and a dictionary mapping landmark IDs to (x, y) pixels.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None")

        try:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Perform detection
            if self.running_mode == vision.RunningMode.VIDEO:
                if timestamp_ms is None:
                    raise ValueError("timestamp_ms is required for VIDEO running mode")
                detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
            else:
                detection_result = self.detector.detect(mp_image)

            landmarks_dict = {}
            ih, iw, _ = frame.shape

            if detection_result.face_landmarks:
                # We process the first detected face for the landmark dictionary
                # to maintain compatibility with existing blink detection logic
                face_lms = detection_result.face_landmarks[0]
                for idx, lm in enumerate(face_lms):
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    landmarks_dict[idx] = (x, y)

                if draw:
                    for face_landmarks in detection_result.face_landmarks:
                        # Draw tessellation
                        drawing_utils.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
                        )
                        # Draw contours
                        drawing_utils.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
                        )

            return frame, landmarks_dict
        except Exception as e:
            raise RuntimeError(f"Error processing frame: {e}")

    def close(self) -> None:
        """Release the detector resources."""
        self.detector.close()


def generate_face_mesh(
    video_path: str | int,
    resizing_factor: float,
    save_video: bool = False,
    filename: Optional[str] = None,
    codec: str = "mp4v",
) -> None:
    """
    Process video stream and generate face mesh visualization.

    Args:
        video_path: Path to video file or 0 for webcam.
        resizing_factor: Factor to resize output display.
        save_video: Boolean to enable video saving.
        filename: Output video filename.
    """
    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(0 if video_path == 0 else video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video capture")

        f_w, f_h, fps = (
            int(cap.get(x))
            for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
        )

        if save_video:
            if not filename:
                raise ValueError("Filename must be provided when save_video is True")

            video_dir = "data/videos/outputs"
            os.makedirs(video_dir, exist_ok=True)
            save_path = os.path.join(video_dir, filename)
            fourcc = cv2.VideoWriter.fourcc(*codec)
            out = cv2.VideoWriter(save_path, fourcc, fps, (f_w, f_h))

        # Use VIDEO mode for better tracking performance in video streams
        mesh_generator = FaceMeshGenerator(running_mode=vision.RunningMode.VIDEO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp in milliseconds
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            frame, _ = mesh_generator.create_face_mesh(frame, draw=True, timestamp_ms=timestamp_ms)

            if save_video and out is not None:
                out.write(frame)

            if video_path == 0:
                frame = cv2.flip(frame, 1)

            if resizing_factor <= 0:
                raise ValueError("Resizing factor must be positive")

            display_w = int(f_w * resizing_factor)
            display_h = int(f_h * resizing_factor)
            resized_frame = cv2.resize(frame, (display_w, display_h))
            cv2.imshow("Face Mesh Detection", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord("p"):
                break

        mesh_generator.close()

    except Exception as e:
        print(f"Error during video processing: {e}")

    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ensure 'face_landmarker.task' is in the project root before running
    input_path = 0
    scale = 1.0 if input_path == 0 else 0.5
    generate_face_mesh(input_path, scale)
