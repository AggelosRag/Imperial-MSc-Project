from PIL import Image, ImageDraw, ImageFont
import textwrap

# Load the digit image
digit_image_path = './images/eight.png'
digit_image = Image.open(digit_image_path)

# Text content
true_class = "True class: 8"
concept_true_values = """
Concept True values:
thickness_small = 0
thickness_medium = 0
thickness_large = 0
thickness_xlarge = 1
width_small = 0
width_medium = 0
width_large = 1
width_xlarge = 0
length_small = 0
length_medium = 1
length_large = 0
length_xlarge = 0
"""

concept_pred_values = """
Concept Predicted values: 
thickness_small = 0.0000000001
thickness_medium = 0.0000186685
thickness_large = 0.0005014005
thickness_xlarge = 0.9998830557
width_small = 0.0000000024
width_medium = 0.0000000029
width_large = 0.9996578693
width_xlarge = 0.0001057763
length_small = 0.0000208666
length_medium = 0.9997848868
length_large = 0.0000332112
length_xlarge = 0.7118660212
"""

decision_path_gt = """
Decision path in ground truth tree:
IF thickness_xlarge = 1   AND 
    width_large = 1   AND
    length_medium = 1
THEN class = 9
"""

decision_path_leaked = """
Decision path in leaked tree: 
IF length_xlarge in (0.03, 0.88) AND
    width_xlarge <= 1.00
THEN class = 8
"""

# Create a white image for the text
dpi = 300
text_image_width = 800
text_image_height = 450
text_image = Image.new('RGB', (text_image_width, text_image_height), color='white')
draw = ImageDraw.Draw(text_image)

# Keywords to be formatted
keywords = {
    "IF": ("orange", "bold"),
    "AND": ("orange", "bold"),
    "THEN": ("blue", "bold"),
    "class = 8": ("green", "bold", "tick"),
    #"class = 9": ("red", "bold", "cross"),
    "= 9": ("red", "bold"),
    "True class: 8": ('black', "bold"),
    "Concept True values:": ('black',"bold"),
    "Concept Predicted values:": ('black',"bold"),
    "Decision path in ground truth tree:": ('black',"bold"),
    "Decision path in leaked tree:": ('black', "bold"),
}

# Use a truetype or opentype font
font_path = "/System/Library/Fonts/Supplemental/Georgia.ttf"  # Update this path to a valid font path
font_size = 16
font = ImageFont.truetype(font_path, font_size)

# Text wrapping and drawing with formatting
margin = 10
offset = 10
tick = "✔"
cross = "✘"

# Function to draw text with formatting
def draw_text_with_formatting(draw, text, font, margin, offset):
    tick = "✔"
    cross = "✘"
    words = text.split(' ')
    for word in words:
        formatted = False
        for key, (color, style, *extra) in keywords.items():
            if key in word:
                if style == "bold":
                    draw.text((margin, offset), word.replace(key, ""), font=font, fill="black")
                    font_style = ImageFont.truetype(font_path, font_size)
                    bbox = draw.textbbox((margin, offset), key, font=font_style)
                    draw.text((margin, offset), key, font=font_style, fill=color)
                    margin += bbox[2] - bbox[0] + 5
                if extra and "tick" in extra:
                    draw.text((margin, offset), tick, font=font_style, fill="green")
                    bbox = draw.textbbox((margin, offset), tick, font=font_style)
                    margin += bbox[2] - bbox[0] + 5
                if extra and "cross" in extra:
                    draw.text((margin, offset), cross, font=font_style, fill="red")
                    bbox = draw.textbbox((margin, offset), cross, font=font_style)
                    margin += bbox[2] - bbox[0] + 5
                formatted = True
                break
        if not formatted:
            draw.text((margin, offset), word, font=font, fill="black")
            bbox = draw.textbbox((margin, offset), word, font=font)
            margin += bbox[2] - bbox[0] + 5
    return margin

# Center the true class text
true_class_bbox = draw.textbbox((0, 0), true_class, font=font)
true_class_width = true_class_bbox[2] - true_class_bbox[0]
draw.text(((text_image_width - true_class_width) / 2, offset), true_class, font=font, fill="black")
offset += true_class_bbox[3] - true_class_bbox[1] + 20  # Add some space after the centered text

# Draw the concept true values and predicted values side by side
true_values_lines = concept_true_values.strip().split('\n')
pred_values_lines = concept_pred_values.strip().split('\n')

max_lines = max(len(true_values_lines), len(pred_values_lines))
line_height = font.getbbox('A')[3]

for i in range(max_lines):
    if i < len(true_values_lines):
        draw_text_with_formatting(draw, true_values_lines[i], font, margin, offset)
    if i < len(pred_values_lines):
        draw_text_with_formatting(draw, pred_values_lines[i], font, text_image_width // 2 + margin, offset)
    offset += line_height + 5

# Add some space before the decision paths
offset += 20

# Split and draw the decision paths side by side
gt_lines = decision_path_gt.strip().split('\n')
leaked_lines = decision_path_leaked.strip().split('\n')

max_lines = max(len(gt_lines), len(leaked_lines))

for i in range(max_lines):
    if i < len(gt_lines):
        draw_text_with_formatting(draw, gt_lines[i], font, margin, offset)
    if i < len(leaked_lines):
        draw_text_with_formatting(draw, leaked_lines[i], font, text_image_width // 2 + margin, offset)
    offset += line_height + 5

# Combine the digit image and text image
combined_width = digit_image.width + text_image.width
combined_height = max(digit_image.height, text_image.height)
combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
combined_image.paste(digit_image, (0, (combined_height - digit_image.height) // 2))
combined_image.paste(text_image, (digit_image.width, 0))

# Save the combined image
combined_image.save('./plots/leaky_eight.png')

# Display the combined image
combined_image.show()
