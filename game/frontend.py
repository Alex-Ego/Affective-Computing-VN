import pygame as game
import time
import os

# Initializing the pygame library
game.init()

# Paths to resources
abs_location = os.path.dirname(os.path.abspath(__file__))
font_location = "/fonts/"

# Screen initialization
game.display.init()
X_size = 1280
Y_size = 720
screen_size = (X_size, Y_size)
screen = game.display.set_mode(screen_size)
game.display.set_caption("Sentiment Analysis beta ver.") 

# Screen color
screen.fill((0,0,0))

# Text Fonts declaration
UI_Text = game.font.Font(abs_location + font_location + "Manjari-Regular.otf", 100)

# Drawing Text
loading_text = UI_Text.render("Loading...", True, (255,255,255))
loading_text_rec = loading_text.get_rect()
loading_text_rec.center = (X_size//2, Y_size//2)
screen.blit(loading_text, loading_text_rec)

# Updating the graphics
game.display.update()

# Wait 10 secs
time.sleep(10)

# Kill process
game.display.quit()