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

# Fonts used and their purpose
class Text_Type():
    def __init__(self, font, x_pos = 640, y_pos = 360):
        self.font = font
        self.x_pos = x_pos
        self.y_pos = y_pos

class Text():
    def __init__(self, screen, text_type, text, bg=(0,0,0), color=(255,255,255), offset_x=0, offset_y=0):
        self.color = color
        self.bg = bg
        
        self.screen = screen
        
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        self.text = text
        self.text_type = text_type
        self.text_label = text_type.font.render(self.text, True, self.color)
        self.text_rec = self.text_label.get_rect()
        self.text_rec.center = (self.text_type.x_pos + self.offset_x, self.text_type.y_pos + self.offset_y)
        
    
    def draw(self, update = False):
        self.screen.blit(self.text_label, self.text_rec)
        if update == True:
            game.display.update()
        
    
    def erase(self, update = False):
        self.screen.fill(self.bg, self.text_rec)
        if update == True:
            game.display.update()

UI_Text = Text_Type(game.font.Font(os.path.join(font_location, "Manjari-Regular.otf"), 100), X_size//2, Y_size//2)