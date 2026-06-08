from PIL import ImageFont, ImageChops, Image
import os

fonts = {
    "SegoePrint": "C:/Windows/Fonts/segoepr.ttf",
    "ComicSans": "C:/Windows/Fonts/comic.ttf"
}

test_chars = ["\u221a", "\u2082", "\u2081", "\u00b2", "\u2212", "d", "="]

for name, path in fonts.items():
    if not os.path.exists(path):
        print(f"{name} path does not exist.")
        continue
    try:
        font = ImageFont.truetype(path, 28)
        mask_notdef = font.getmask("\uffff")
        print(f"Font: {name}")
        for char in test_chars:
            mask = font.getmask(char)
            is_missing = False
            if mask.size == mask_notdef.size:
                img1 = Image.frombytes("L", mask.size, bytes(mask))
                img2 = Image.frombytes("L", mask_notdef.size, bytes(mask_notdef))
                diff = ImageChops.difference(img1, img2)
                if diff.getbbox() is None:
                    is_missing = True
            char_hex = hex(ord(char))
            print(f"  Character {char_hex}: {'MISSING (will render box)' if is_missing else 'AVAILABLE'}")
    except Exception as e:
        print(f"  Error testing {name}: {e}")
    print("-" * 30)
