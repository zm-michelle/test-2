from pathlib import Path

from datasets.simulated_noisy_office import list_simulated_noisy_office_images
from ocr_simulated_noisy_office import OCRLine, reconstruct_text, sort_ocr_lines
from utils import safe_relative_txt_path


def test_sort_ocr_lines_top_to_bottom_left_to_right():
    lines = [
        OCRLine([[50, 30], [60, 30], [60, 40], [50, 40]], "b", 0.9),
        OCRLine([[10, 10], [20, 10], [20, 20], [10, 20]], "a", 0.9),
        OCRLine([[10, 30], [20, 30], [20, 40], [10, 40]], "c", 0.9),
    ]
    assert [line.text for line in sort_ocr_lines(lines)] == ["a", "c", "b"]


def test_reconstruct_text():
    assert reconstruct_text([OCRLine([], "one", 1.0), OCRLine([], "two", 1.0)]) == "one\ntwo"


def test_safe_relative_txt_path_sanitizes():
    root = Path("/tmp/root")
    image = root / "clean_images_binaryscal (low resolution)" / "sample.png"
    assert safe_relative_txt_path(image, root).as_posix() == "clean_images_binaryscal__low_resolution/sample.txt"


def test_list_simulated_noisy_office_images_limit(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.png").write_bytes(b"")
    (tmp_path / "a" / "two.txt").write_text("nope")
    assert len(list_simulated_noisy_office_images(tmp_path, limit=1)) == 1

