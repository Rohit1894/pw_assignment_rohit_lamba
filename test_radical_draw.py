from PIL import Image, ImageDraw, ImageFont
import os
import re

font_path = "C:/Windows/Fonts/Inkfree.ttf"
font = ImageFont.truetype(font_path, 28)

def draw_custom_text(draw, x, y, text, font, color):
    """Draw text character-by-character, mapping subscripts/superscripts/minus signs."""
    curr_x = x
    try:
        font_path = getattr(font, "path", None)
        if font_path and os.path.exists(font_path):
            sub_font = ImageFont.truetype(font_path, max(10, int(font.size * 0.65)))
        else:
            sub_font = font
    except Exception:
        sub_font = font

    SUPERSCRIPTS_MAP = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }
    
    for char in text:
        char_to_draw = char
        curr_font = font
        curr_y = y
        
        if char == '−': # Unicode minus
            char_to_draw = '-'
        elif '₀' <= char <= '₉':
            char_to_draw = str(ord(char) - 0x2080)
            curr_font = sub_font
            curr_y = y + int(font.size * 0.25)
        elif char in SUPERSCRIPTS_MAP:
            char_to_draw = SUPERSCRIPTS_MAP[char]
            curr_font = sub_font
            curr_y = y - int(font.size * 0.15)
            
        draw.text((curr_x, curr_y), char_to_draw, fill=color, font=curr_font)
        curr_x += draw.textlength(char_to_draw, font=curr_font)
        
    return curr_x - x


def get_custom_text_width(draw, text, font):
    """Calculate the width of the text using custom sub-font/subscript mapping."""
    curr_x = 0
    try:
        font_path = getattr(font, "path", None)
        if font_path and os.path.exists(font_path):
            sub_font = ImageFont.truetype(font_path, max(10, int(font.size * 0.65)))
        else:
            sub_font = font
    except Exception:
        sub_font = font

    SUPERSCRIPTS_MAP = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }
    
    for char in text:
        char_to_draw = char
        curr_font = font
        
        if char == '−': # Unicode minus
            char_to_draw = '-'
        elif '₀' <= char <= '₉':
            char_to_draw = str(ord(char) - 0x2080)
            curr_font = sub_font
        elif char in SUPERSCRIPTS_MAP:
            char_to_draw = SUPERSCRIPTS_MAP[char]
            curr_font = sub_font
            
        curr_x += draw.textlength(char_to_draw, font=curr_font)
        
    return curr_x


def _draw_handwritten_line(draw, x1, y1, x2, y2, width=3, color=(0,0,0)):
    # simple straight line for testing
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)


def draw_math_equation_with_radicals(draw, x, y, text, font, color):
    if "√" not in text:
        draw_custom_text(draw, x, y, text, font, color)
        return
        
    parts = text.split("√")
    curr_x = x
    
    for idx, part in enumerate(parts):
        if idx == 0:
            if part:
                curr_x += draw_custom_text(draw, curr_x, y, part, font, color)
        else:
            if part.startswith("("):
                depth = 0
                closing_idx = -1
                for char_idx, char in enumerate(part):
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        depth -= 1
                        if depth == 0:
                            closing_idx = char_idx
                            break
                if closing_idx != -1:
                    inside = part[1:closing_idx]
                    rest = part[closing_idx+1:]
                else:
                    inside = part[1:]
                    rest = ""
            else:
                match = re.match(r'^[0-9]+', part)
                if match:
                    inside = match.group(0)
                    rest = part[len(inside):]
                else:
                    inside = part
                    rest = ""
                    
            inside_w = get_custom_text_width(draw, inside, font) if inside else 0
            
            r_width = 2
            rx0 = curr_x
            ry0 = y + 15
            
            rx1 = curr_x + 6
            ry1 = y + 19
            
            rx2 = curr_x + 14
            ry2 = y + 30
            
            rx3 = curr_x + 22
            ry3 = y - 4
            
            rx4 = curr_x + 22 + int(inside_w) + 2
            ry4 = y - 4
            
            _draw_handwritten_line(draw, rx0, ry0, rx1, ry1, r_width, color)
            _draw_handwritten_line(draw, rx1, ry1, rx2, ry2, r_width, color)
            _draw_handwritten_line(draw, rx2, ry2, rx3, ry3, r_width, color)
            _draw_handwritten_line(draw, rx3, ry3, rx4, ry4, r_width, color)
            
            if inside:
                draw_custom_text(draw, curr_x + 24, y, inside, font, color)
                curr_x += 24 + inside_w + 6
                
            if rest:
                curr_x += draw_custom_text(draw, curr_x, y, rest, font, color)

img = Image.new("RGB", (800, 300), (255, 255, 255))
draw = ImageDraw.Draw(img)
draw_math_equation_with_radicals(draw, 10, 20, "d = √((x₂−x₁)² + (y₂−y₁)²)", font, (0,0,0))
draw_math_equation_with_radicals(draw, 10, 100, "d = √((4−1)² + (6−2)²)", font, (0,0,0))
draw_math_equation_with_radicals(draw, 10, 180, "d = √(9 + 16) = √25", font, (0,0,0))

os.makedirs("output/analysis", exist_ok=True)
img.save("output/analysis/test_radical_handdrawn.png")
print("Saved test radical draw successfully!")
