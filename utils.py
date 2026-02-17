"""Utility functions for drawing operations in OpenCV with enhanced error handling."""

import cv2 as cv
import numpy as np


import os
from pathlib import Path
from typing import Any, Literal, overload


class DrawingUtils:
    """Utility class for OpenCV drawing operations with enhanced error handling."""

    @staticmethod
    def draw_overlay(
        frame: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        alpha: float = 0.25,
        color: tuple[int, int, int] = (51, 68, 255),
        *,
        filled: bool = True,
    ) -> None:
        """Draw a semi-transparent overlay on the frame.

        Args:
            frame (np.ndarray): Input image
            pt1 (tuple): Top-left corner coordinates
            pt2 (tuple): Bottom-right corner coordinates
            alpha (float): Transparency level (0-1)
            color (tuple): Rectangle color (B,G,R)
            filled (bool): Whether to fill the rectangle

        Raises:
            ValueError: If input parameters are invalid
            TypeError: If input types are incorrect
        """
        # Input validation
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy array")

        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")

        try:
            overlay = frame.copy()
            rect_color = color if filled else (0, 0, 0)
            cv.rectangle(overlay, pt1, pt2, rect_color, cv.FILLED if filled else 1)
            cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        except Exception as e:
            raise RuntimeError(f"Error drawing overlay: {e!s}") from e

    @staticmethod
    def draw_rounded_rect(
        img: np.ndarray,
        bbox: tuple[int, int, int, int] | list,
        line_color: tuple[int, int, int] = (255, 255, 255),
        ellipse_color: tuple[int, int, int] = (0, 0, 255),
        line_thickness: int = 2,
        ellipse_thickness: int = 3,
        radius: int = 15,
    ) -> None:
        """Draw a rectangle with rounded corners.

        Args:
            img (np.ndarray): Input image
            bbox (tuple/list): Bounding box coordinates (x1, y1, x2, y2)
            line_color (tuple[int, int, int]): Color for straight lines
            ellipse_color (tuple[int, int, int]): Color for corner ellipses
            line_thickness (int): Thickness of straight lines
            ellipse_thickness (int): Thickness of corner ellipses
            radius (int): Radius of corner rounding

        Raises:
            ValueError: If input parameters are invalid
            TypeError: If input types are incorrect
        """
        # Input validation
        if not isinstance(img, np.ndarray):
            raise TypeError("Image must be a numpy array")

        if len(bbox) != 4:
            raise ValueError("Bounding box must contain 4 coordinates")

        x1, y1, x2, y2 = bbox

        try:
            # Draw straight lines
            cv.line(img, (x1 + radius, y1), (x2 - radius, y1), line_color, line_thickness)
            cv.line(img, (x1 + radius, y2), (x2 - radius, y2), line_color, line_thickness)
            cv.line(img, (x1, y1 + radius), (x1, y2 - radius), line_color, line_thickness)
            cv.line(img, (x2, y1 + radius), (x2, y2 - radius), line_color, line_thickness)

            # Draw corner ellipses
            corner_points = [
                ((x1 + radius, y1 + radius), 180),
                ((x2 - radius, y1 + radius), 270),
                ((x1 + radius, y2 - radius), 90),
                ((x2 - radius, y2 - radius), 0),
            ]

            for center, angle in corner_points:
                cv.ellipse(
                    img, center, (radius, radius), angle, 0, 90, ellipse_color, ellipse_thickness
                )

        except Exception as e:
            raise RuntimeError(f"Error drawing rounded rectangle: {e!s}") from e

    @staticmethod
    def draw_text_with_bg(
        frame: np.ndarray,
        text: str,
        pos: tuple[int, int],
        font: int = cv.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.3,
        thickness: int = 1,
        bg_color: tuple[int, int, int] = (255, 255, 255),
        text_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """Draw text with a background rectangle.

        Args:
            frame (np.ndarray): Input image
            text (str): Text to draw
            pos (tuple[int, int]): Starting position of text
            font (int): OpenCV font type
            font_scale (float): Font scale factor
            thickness (int): Line thickness
            bg_color (tuple[int, int, int]): Background rectangle color
            text_color (tuple[int, int, int]): Text color

        Raises:
            ValueError: If input parameters are invalid
            TypeError: If input types are incorrect
        """
        # Input validation
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy array")

        if not text:
            raise ValueError("Text cannot be empty")

        try:
            # Calculate text size
            (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
            x, y = pos

            # Draw background rectangle
            cv.rectangle(
                frame,
                (x, y - text_height - baseline),
                (x + text_width, y + baseline),
                bg_color,
                cv.FILLED,
            )

            # Draw text
            cv.putText(
                frame, text, (x, y), font, font_scale, text_color, thickness, lineType=cv.LINE_AA
            )

        except Exception as e:
            raise RuntimeError(f"Error drawing text with background: {e!s}") from e


@overload
def find_files(
    folder: str | Path,
    *,
    extensions: str | list[str] | None = None,
    name_patterns: str | list[str] | None = None,
    depth: int | None = None,
    case_sensitive: bool = False,
    followlinks: bool = True,
    as_path: Literal[False] = False,
) -> list[str]: ...


@overload
def find_files(
    folder: str | Path,
    *,
    extensions: str | list[str] | None = None,
    name_patterns: str | list[str] | None = None,
    depth: int | None = None,
    case_sensitive: bool = False,
    followlinks: bool = True,
    as_path: Literal[True],
) -> list[Path]: ...


def find_files(
    folder: str | Path,
    *,
    extensions: str | list[str] | None = None,
    name_patterns: str | list[str] | None = None,
    depth: int | None = None,
    case_sensitive: bool = False,
    followlinks: bool = True,
    as_path: bool = False,
) -> list[str] | list[Path]:
    """Recursively finds all files matching given extensions and/or name patterns.

    This function traverses the directory tree and returns a sorted list of absolute file paths
    that satisfy the search criteria. At least one of `extensions` or `name_patterns` must be
    provided.

    Args:
        folder (str | Path): Path to the folder to search.
        extensions (str | list[str] | None, optional): A single file extension (e.g., ".txt")
            or a list of extensions to search for (e.g., [".jpg", ".png"]).
        name_patterns (str | list[str] | None, optional): A single glob-style pattern
            (e.g., "report_*") or a list of patterns to match against the full filename.
            The `*` wildcard is supported.
        depth (int | None, optional): The maximum depth to recurse.
            0 means only the root folder. None or -1 means unlimited depth. Defaults to None.
        case_sensitive (bool, optional): If True, the extension match is
            case-sensitive. Defaults to False (e.g., ".jpg" will match ".JPG").
        followlinks (bool, optional): If True, the search will traverse into directories that are
            symbolic links. Defaults to True.
        as_path (bool, optional): If True, returns a list of Path objects instead of strings.

    Returns:
        (list[str] | list[Path]): A sorted list of absolute file paths matching the criteria.
    """
    if extensions is None and name_patterns is None:
        raise ValueError("At least one of `extensions` or `name_patterns` must be provided.")

    folder_path = Path(folder).resolve()
    found_files: list[Any] = []

    # --- Normalize Extensions ---
    ext_set = set()
    if extensions:
        ext_list = [extensions] if isinstance(extensions, str) else extensions
        for ext in ext_list:
            normalized_ext = f".{ext.lstrip('.')}"
            ext_set.add(normalized_ext if case_sensitive else normalized_ext.lower())

    # --- Normalize Name Patterns ---
    pattern_list = []
    if name_patterns:
        pattern_list = [name_patterns] if isinstance(name_patterns, str) else name_patterns
        if not case_sensitive:
            pattern_list = [p.lower() for p in pattern_list]

    # --- Walk the Directory Tree using os.walk for symlink control ---
    for dirpath, dirnames, filenames in os.walk(str(folder_path), followlinks=followlinks):
        # --- Depth Control ---
        if depth is not None and depth >= 0:
            # Calculate current depth relative to the root folder
            try:
                rel_path = Path(dirpath).relative_to(folder_path)
                if str(rel_path) == ".":
                    current_depth = 0
                else:
                    current_depth = len(rel_path.parts)

                if current_depth > depth:
                    del dirnames[:]
                    continue
                if current_depth == depth:
                    del dirnames[:]
            except ValueError:
                # Should not happen if walking from folder_path
                continue

        for filename in filenames:
            # Create a full Path object for matching and storage
            f = Path(dirpath) / filename

            # Prepare file attributes for comparison
            file_name_to_check = f.name if case_sensitive else f.name.lower()
            file_suffix_to_check = f.suffix if case_sensitive else f.suffix.lower()

            # --- Apply Filters ---
            passes_ext_filter = not ext_set or file_suffix_to_check in ext_set

            passes_name_filter = not pattern_list or any(
                # Use pathlib's match for glob patterns
                Path(file_name_to_check).match(p)
                for p in pattern_list
            )

            if passes_ext_filter and passes_name_filter:
                if as_path:
                    found_files.append(f.resolve())
                else:
                    found_files.append(str(f.resolve()))

    return sorted(found_files)


# Example usage
def main() -> None:
    """Demonstrate the DrawingUtils functions with error handling."""
    try:
        # Create a sample image
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Demonstrate drawing methods with error handling
        DrawingUtils.draw_overlay(img, (50, 50), (200, 200))
        DrawingUtils.draw_rounded_rect(img, (100, 100, 250, 250))
        DrawingUtils.draw_text_with_bg(img, "Hello, OpenCV!", (50, 50))

        cv.imshow("Drawing Utilities Demo", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
