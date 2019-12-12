# The script of the game goes in this file.

# Declare characters used by this game. The color argument colorizes the
# name of the character.

define t = Character("Temp")


# The game starts here.

label start:

    # Show a background. This uses a placeholder by default, but you can
    # add a file (named either "bg room.png" or "bg room.jpg") to the
    # images directory to show it.

    scene sky2

    # These display lines of dialogue.
    $ print(dictionary)
    while 1:
        show assistant neutral
        t "Let's try this out."
        python:
            userinput = renpy.input("Say something.")
            phraseanalysis(userinput, dictionary, w)

    # This ends the game.

    return
