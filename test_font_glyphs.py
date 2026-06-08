from PIL import Image, ImageFont
import os

fonts = {
    "Inkfree": "C:/Windows/Fonts/Inkfree.ttf",
    "SegoePrint": "C:/Windows/Fonts/segoepr.ttf",
    "ComicSans": "C:/Windows/Fonts/comic.ttf",
    "Arial": "C:/Windows/Fonts/arial.ttf"
}

test_chars = ["\u221a", "\u2082", "\u2081", "\u00b2", "d", "=", "x"]

for name, path in fonts.items():
    if not os.path.exists(path):
        print(f"{name} path does not exist.")
        continue
    try:
        font = ImageFont.truetype(path, 24)
        print(f"Font: {name}")
        for char in test_chars:
            try:
                mask = font.getmask(char)
                bbox = mask.getbbox()
                char_hex = hex(ord(char))
                print(f"  Character {char_hex}: {'Supported' if bbox else 'MISSING'}")
            except Exception as ex:
                print(f"  Error testing char: {ex}")
    except Exception as e:
        print(f"  Error testing {name}: {e}")
    print("-" * 30)
