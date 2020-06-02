import time
import VNCommands as VN
import sentiment_analysis as SA

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
time.sleep(5)

conversation = VN.Scene("sky2.png", "assistant neutral.png", VN.screen)
conversation.say("Alright, let's do this one more time.")
time.sleep(3)
conversation.say("Are you ready?", "assistant thumbs up.png")
time.sleep(3)
# prompt = VN.Text(VN.screen, VN.UI_Text, "Type in console.", pos = VN.position.center)
while True:
    events = VN.game.event.get()
    for event in events:
        if event.type == VN.game.QUIT:
            exit()
    conversation.say("", "assistant neutral.png")
    conversation.say("Write on console.")
    userinput = input("Say something: ")
    if userinput:
        score = SA.evaluation(userinput)
        if score == "happiness":
            conversation.say("Hey, that's pretty good!", "assistant thumbs up.png")
        if score == "neutral":
            conversation.say("Ah, is that so?")
        if score == "sadness":
            conversation.say("Don't be sad!", "assistant sad.png")

    time.sleep(3)