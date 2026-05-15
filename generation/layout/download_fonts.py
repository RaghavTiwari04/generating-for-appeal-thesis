"""Download Google Fonts needed by font_palette.py into generation/layout/fonts/.

Fonts are licensed under SIL OFL and are safe for commercial use.
Run once at setup: `python -m generation.layout.download_fonts`
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx

FONTS_DIR = Path(__file__).parent / "fonts" / "google"
FONTS_DIR.mkdir(parents=True, exist_ok=True)

# (dest_filename, google_fonts_api_url_or_direct_download)
FONT_DOWNLOADS: list[tuple[str, str]] = [
    ("Caveat-Bold.ttf",         "https://fonts.gstatic.com/s/caveat/v18/Wnz6HAc5bAfYB2QRah7pcpNvOx-pjfJ9eIWpZA.ttf"),
    ("Sacramento-Regular.ttf", "https://fonts.gstatic.com/s/sacramento/v14/buEzpo6gcdjy0EiZMBUG0Co.ttf"),
    ("CormorantGaramond-SemiBold.ttf", "https://fonts.gstatic.com/s/cormorantgaramond/v21/co3YmX5slCNuHLi8bLeY9MK7whWMhyjYrEtjEtfA.ttf"),
    ("Spectral-SemiBold.ttf",  "https://fonts.gstatic.com/s/spectral/v13/rnCu-xNNww_2s0amA9v2s1zRmbku.ttf"),
    ("BebasNeue-Regular.ttf",  "https://fonts.gstatic.com/s/bebasneue/v14/JTUSjIg69CK48gW7PXooxW5rygbi49c.ttf"),
    ("Anton-Regular.ttf",      "https://fonts.gstatic.com/s/anton/v25/1Ptgg87LROyAm0K08i4gS7lu.ttf"),
    ("PlayfairDisplay-Bold.ttf", "https://fonts.gstatic.com/s/playfairdisplay/v37/nuFvD-vYSZviVYUb_rj3ij__anPXJzDwcbmjWBN2PKdFvUDQZNLo_U2r.ttf"),
    ("Lora-SemiBold.ttf",      "https://fonts.gstatic.com/s/lora/v35/0QI6MX1D_JOxE7fSiH-f6Pzjp2hxxWWNEQ.ttf"),
    ("Rubik-SemiBold.ttf",     "https://fonts.gstatic.com/s/rubik/v28/iJWZBXyIfDnIV5PNhY1KTN7Z-Yh-B4iFWkU1Z4Y.ttf"),
    ("AmaticSC-Bold.ttf",      "https://fonts.gstatic.com/s/amaticsc/v26/TUZyzwprpvBS1izr_vO0De6ecZQf1A.ttf"),
    ("PermanentMarker-Regular.ttf", "https://fonts.gstatic.com/s/permanentmarker/v16/Fh4uPib9Iyv2ucM6pGQMWimMp004La2Cf5b6jlg.ttf"),
    ("GreatVibes-Regular.ttf", "https://fonts.gstatic.com/s/greatvibes/v19/RWmMoKWR9v4ksMfaWd_JN9XFiaQ.ttf"),
    ("DancingScript-Bold.ttf", "https://fonts.gstatic.com/s/dancingscript/v25/If2cXTr6YS-zF4S-kcSWSVi_sxjsohD9F50Ruu7BMSo3Sup6.ttf"),
    ("AlfaSlabOne-Regular.ttf","https://fonts.gstatic.com/s/alfaslabone/v19/6NUQ8FmMKwSEKjnm5-4v-4Jh6dVretWvYmE.ttf"),
    ("Righteous-Regular.ttf",  "https://fonts.gstatic.com/s/righteous/v17/1cXxaUPXBpj2rGoU7C9mj3uEicG01A.ttf"),
]


def download_all(force: bool = False) -> None:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for filename, url in FONT_DOWNLOADS:
            dest = FONTS_DIR / filename
            if dest.exists() and not force:
                print(f"  skip  {filename} (already downloaded)")
                continue
            print(f"  fetch {filename} ...", end=" ", flush=True)
            try:
                resp = client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                print(f"OK ({len(resp.content)//1024} KB)")
            except Exception as e:
                print(f"FAILED: {e}")

    manifest = FONTS_DIR / "LICENSE_SIL_OFL.txt"
    if not manifest.exists():
        manifest.write_text(
            "All fonts in this directory are licensed under the SIL Open Font License v1.1.\n"
            "https://scripts.sil.org/OFL\n"
        )
    print(f"Fonts saved to {FONTS_DIR}")


if __name__ == "__main__":
    import typer

    typer.run(download_all)
