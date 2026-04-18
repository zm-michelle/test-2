from pathlib import Path

from datasets.sroie import parse_sroie_annotation


def test_parse_sroie_annotation(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("1,2,3,4,5,6,7,8,HELLO, WORLD\nbad,line\n", encoding="utf-8")
    boxes = parse_sroie_annotation(path)
    assert len(boxes) == 1
    assert boxes[0].text == "HELLO, WORLD"
    assert boxes[0].points[0] == (1, 2)


def test_parse_sroie_annotation_tolerates_unclosed_quotes(tmp_path: Path):
    path = tmp_path / "sample.csv"
    path.write_text(
        '46,2901,720,2901,720,2928,46,2928,"EDEEMED WITHIN THREE(3) MONTHS\n'
        "48,2933,535,2933,535,2964,48,2964,INVOICE.(TERMS & CONDITIONS APPLIED)\n",
        encoding="utf-8",
    )
    boxes = parse_sroie_annotation(path)
    assert len(boxes) == 2
    assert boxes[0].text == "EDEEMED WITHIN THREE(3) MONTHS"
    assert boxes[1].text == "INVOICE.(TERMS & CONDITIONS APPLIED)"
