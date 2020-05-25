import time
import VNCommands as VN

# Screen color
VN.screen.fill((0,0,0))

# Drawing Text
Test = VN.Text(VN.screen, VN.UI_Text, "Loading...")
Test.draw()

# Updating the graphics
VN.game.display.update()

# Declaring the text label
loading_progress = VN.Text(VN.screen, VN.UI_Text, "", pos = VN.position.center, offset_y = 200)

for progress in range(101):
    loading_progress.refresh(str(progress) + "%", update = True)
    time.sleep(.05)
    
time.sleep(3)
loading_progress.refresh("Done!", update = True)
time.sleep(3)

sky_bg = VN.Image(VN.screen, "sky2.png", pos = VN.position.center)
assistant = VN.Image(VN.screen, "assistant neutral.png", pos = VN.position.center)
time.sleep(1)
sky = VN.Scene(sky_bg, assistant)
dialogue = VN.Text(VN.screen, VN.Dialogue_Text, "Alright, let's do this one more time.", offset_y = 250)
sky.say(dialogue)
time.sleep(3)
dialogue.refresh("Are you ready?")
sky.say(dialogue)
time.sleep(3)
# prompt = VN.Text(VN.screen, VN.UI_Text, "Type in console.", pos = VN.position.center)
while True:
    events = VN.game.event.get()
    for event in events:
        if event.type == VN.game.QUIT:
            exit()
    dialogue.refresh("")
    sky.say(dialogue)
    userinput = input("Say something: ")
    if userinput:
        dialogue.refresh("Hey, I heard that!")
        sky.say(dialogue)
    #dialogue.refresh("What the fuck")
    #sky.say(dialogue)
    time.sleep(3)