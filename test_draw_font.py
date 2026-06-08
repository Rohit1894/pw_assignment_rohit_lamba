from PIL import Image, ImageDraw, ImageFont
import os

font_path = "C:/Windows/Fonts/Inkfree.ttf"
if os.path.exists(font_path):
    print("Inkfree.ttf exists!")
    try:
        font = ImageFont.truetype(font_path, 28)
        img = Image.new("RGB", (600, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Using exact unicode characters from annotations
        draw.text((10, 10), "d = \u221a((x\u2082\u2212x\u2081)\u00b2 + (y\u2082\u2212y\u2081)\u00b2)", fill=(0,0,0), font=font)
        os.makedirs("output/analysis", exist_ok=True)
        img.save("output/analysis/test_render_inkfree.png")
        print("Successfully rendered inkfree test!")
    except Exception as e:
        print(f"Error drawing with Inkfree: {e}")
else:
    print("Inkfree.ttf does not exist at path.")
