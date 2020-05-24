import time
import VNCommands as VN

# Screen color
VN.screen.fill((0,0,0))

# Drawing Text
#VN.drawtext(VN.screen, VN.UI_Text, "Hope this works")
Test = VN.Text(VN.screen, VN.UI_Text, "Loading...")
Test.draw()

# Updating the graphics
VN.game.display.update()

for progress in range(101):
    loading_progress = VN.Text(VN.screen, VN.UI_Text, str(progress) + "%", offset_y = 200)
    loading_progress.draw(True)
    time.sleep(.2)
    loading_progress.erase()

loading_progress = VN.Text(VN.screen, VN.UI_Text, "Done!", offset_y = 200)
loading_progress.draw(True)
time.sleep(3)

sky_bg = VN.game.image.load(VN.os.path.join(VN.image_location, "sky2.png"))

VN.game.display.update()
time.sleep(5)

VN.game.display.quit()