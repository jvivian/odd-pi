from odd_pi.core import run_bot
from typing import Optional
from pathlib import Path
import polars as pl
from rich import print

import typer

app = typer.Typer()
pl.Config.set_tbl_rows(80)


@app.command()
def list_classes(filt_str: Optional[str] = None):
    """List YOLO classes filtered by optional substr"""
    names = Path(__file__).parent.parent / "data" / "yolo.names"
    df = pl.read_csv(names, has_header=False, new_columns=["class"]).sort("class")
    if filt_str:
        df = df.filter(pl.col("class").str.contains(filt_str))
    print(df["class"])


@app.command()
def run_discord_bot(
    run_active_cam: bool = True,  # Runs camera at interval looking for a class to identify
    cam_interval: int = 30,  # Number of seconds to camera checks for target class
    cam_class: str = "cat",  # Object to identify. Run `list_classes` to see all options
    cam_channel: str = "general",  # Disord channel to post active cam pictures to
    pic_keyword: str = "pi!",  # Command a user types in a Discord channel to take a picture
    yolo_keyword: str = "yolo!",  # Same as above but with the YOLO predictions overlayed
    pic_dir: str = "./pics",  # Intermediate directory for pictures
):
    """Run Discord bot that communicates with the Raspberry Pi camera"""
    run_bot(**locals())


if __name__ == "__main__":
    app()
