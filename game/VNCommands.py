import pygame as game
import os

game.init()
game.display.init()

X_size = 1280
Y_size = 720

screen_size = (X_size, Y_size)
screen = game.display.set_mode(screen_size)

game.display.set_caption("Sentiment Analysis beta ver.") 

# Resource location
abs_location = os.path.dirname(os.path.abspath(__file__)) # Absolute location
font_location = os.path.join(abs_location, "fonts")
image_location = os.path.join(abs_location, "images")

# Positions
# List of predetermined positions, these can be offsetted as needed
class position:
    top_left = (0, 0)
    top_center = (X_size//2, 0)
    top_right = (X_size, 0)
    center_left = (0, Y_size//2)
    center = (X_size//2, Y_size//2)
    center_right = (X_size, Y_size//2)
    bottom_left = (0, Y_size)
    bottom_center = (X_size//2, Y_size)
    bottom_right = (X_size, Y_size)

# Fonts used and their purpose
class Text_Type():
    def __init__(self, font, size=50):
        self.font = font
        self.size = size
        self.font_format = game.font.Font(os.path.join(font_location, font), size)

class Text():
    def __init__(self, screen, text_type, text, bg=(0,0,0), color=(255,255,255), pos = position.center, offset_x = 0, offset_y = 0):
        self.color = color      # Text Color
        self.bg = bg            # Background Color
        
        self.screen = screen    # Screen where it'll be displayed
        self.pos = pos
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        self.text = text
        self.text_type = text_type
        self.text_label = text_type.font_format.render(self.text, True, self.color)
        self.text_rec = self.text_label.get_rect()
        self.text_rec.center = (self.pos[0] + self.offset_x, self.pos[1] + self.offset_y)
        
    # Draws text
    def draw(self, bg = False, update = False):
        # If bg is true, the text will have a background
        if bg == True:
            self.screen.fill(self.bg, self.text_rec)
        
        self.screen.blit(self.text_label, self.text_rec)
        
        # If update is true, updates the screen automatically, use when it's the only function appearing on screen at the time.
        if update == True:
            game.display.update()
        
    # Erases the text
    def erase(self, update = False):
        self.screen.fill(self.bg, self.text_rec)
        # If update is true, updates the screen automatically, use when it's the only function appearing on screen at the time.
        if update == True:
            game.display.update()
    
    # Replaces previous text with new in the same label
    def refresh(self, new_text, update = False):
        self.erase()
        self.text = new_text
        self.text_label = self.text_type.font_format.render(self.text, True, self.color)
        self.text_rec = self.text_label.get_rect()
        self.text_rec.center = (self.pos[0] + self.offset_x, self.pos[1] + self.offset_y)
        self.draw()
        # If update is true, updates the screen automatically, blabla, you get it.
        if update == True:
            game.display.update()

class Image():
    def __init__(self, screen, image, bg = (0,0,0), pos = position.center, offset_x = 0, offset_y = 0):
        self.image = game.image.load(os.path.join(image_location, image))
        
        self.bg = bg            # Background Color
        
        self.screen = screen
        self.pos = pos
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.image_rec = self.image.get_rect()
        self.image_rec.center = (self.pos[0] + self.offset_x, self.pos[1] + self.offset_y)
        
    # Draws the image
    def draw(self, update = False):
        self.screen.blit(self.image, self.image_rec)
        # If update is true, updates the screen automatically, use when it's the only function appearing on screen at the time.
        if update == True:
            game.display.update()
        
    # Erases the image and replaces it with the bg color
    def erase(self, update = False):
        self.screen.fill(self.bg, self.image_rec)
        # If update is true, updates the screen automatically, use when it's the only function appearing on screen at the time.
        if update == True:
            game.display.update()
    
    # Replaces previous image with new in the same label
    def refresh(self, new_image, update = False):
        self.erase()
        self.image = game.image.load(os.path.join(image_location, new_image))
        self.image_rec = self.image.get_rect()
        self.image_rec.center = (self.pos[0] + self.offset_x, self.pos[1] + self.offset_y)
        self.draw()
        # If update is true, updates the screen automatically, blabla, you get it.
        if update == True:
            game.display.update()

class Scene():
    def __init__(self, bg, actor):
        self.bg = bg
        self.actor = actor
    
    def say(self, dialogue, new_actor = None):
        self.bg.draw()
        if new_actor is not None:
            self.actor = new_actor
        self.actor.draw()
        self.dialogue = dialogue
        self.dialogue.draw(bg = True)
        game.display.update()

# Types of fonts being used in this whole project
UI_Text = Text_Type("Manjari-Regular.otf", 100)
Dialogue_Text = Text_Type("Manjari-Regular.otf", 50)