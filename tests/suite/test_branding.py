from app.reports.branding import MARKER, brand
from app.reports import theme


def test_brand_once_and_idempotent():
    h = "<html><head></head><body><p>x</p></body></html>"
    b = brand(h)
    assert b.count(MARKER) >= 1 and "T58" in b and b.index("T58") < b.rindex("</body>")
    assert brand(b) == b


def test_brand_skips_fragments_and_non_strings():
    assert brand("<p>fragment</p>") == "<p>fragment</p>"
    assert brand(None) is None


def test_theme_page_is_branded():
    assert MARKER in theme.page("T", "<p>b</p>")


def test_writers_brand_output(tmp_path):
    from app.quant_lab.vol_surface import render_surface_html  # noqa: F401  (import check)
    from app.reports.branding import brand_file
    f = tmp_path / "r.html"
    f.write_text("<html><head></head><body>r</body></html>")
    brand_file(f)
    brand_file(f)
    assert f.read_text().count('class="t58-report-mark"') == 1
