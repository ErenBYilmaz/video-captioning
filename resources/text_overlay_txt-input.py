from PIL import Image, ImageFont, ImageDraw

my_image = Image.open("/Users/ronaldseidel/PycharmProjects/venv_waterkant.coding23/data/Frame at 00-00-16.jpg")

foreground = Image.open("/Users/ronaldseidel/PycharmProjects/venv_waterkant.coding23/data/Frame at 00-00-16_overlay-black-background.png")
my_image.paste(foreground, (0, 0), foreground)

font_trivial = ImageFont.truetype('/Users/ronaldseidel/PycharmProjects/venv_waterkant.coding23/data/Arial Black.ttf', 50)
font_species = ImageFont.truetype('/Users/ronaldseidel/PycharmProjects/venv_waterkant.coding23/data/Arial Italic.ttf', 31)
font_facts = ImageFont.truetype('/Users/ronaldseidel/PycharmProjects/venv_waterkant.coding23/data/Arial.ttf', 20)



trivial = "Tigershark"
species = "Galeocerdo cuvier"
facts = "- ground shark and large macropredator \n\n- solitary, mostly nocturnal hunter \n\n- widest food spectrum of all sharks \n\n\nLength: 3-6m \n\nWeight: 175-900kg \n\nDiet: fish, birds, squid, turtles, sea snakes, \n    dolphins – and garbage \n\nHabitat: close to the coast, mainly in \n    tropical and subtropical waters"

image_editable = ImageDraw.Draw(my_image)
image_editable.text((80,15), trivial, (237, 230, 211), font=font_trivial)
image_editable.text((140,80), species, (237, 230, 211), font=font_species)
image_editable.text((80,155), facts, (237, 230, 211), font=font_facts)


my_image.save("frame_with_text_from_pillow2.jpg")