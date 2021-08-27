# Sentiment Analysis through a chatbot

This consists in a chatbot that will recognize patterns in an input to determine whoever's writing it feeling, and acting accordingly.
Still a WIP.

## Contents

This repository has all the files needed to run the algorithm, and the thesis itself as well.
- [thesis][LiThe]: Contains the LaTeX files.
- [game][LiGame]: Contains all the files needed to run it, such as Python scripts and datasets (Formerly tried using Ren'py, still needs cleaning)
- [design][LiDesign]: All the design attempts at modeling an assistant are in this folders.

### Libraries used in this project

| Library | Site | Version |
| ------ | ------ | ------ |
| pygame | [Link][PkPG] | 2.0.0 |
| Keras | [Link][PkKe] | 2.4.3 |
| TensorFlow | [Link][PkTF] | 2.5.0 |
| nltk | [Link][PkNLTK] | 3.5 |

### How to Use

```bash
# Running the frontend, should be straight-forward.
$ python3 frontend.py

# In case you need to retrain it.
$ python3 sentiment_analysis_training.py
```
 [LiThe]: <https://github.com/Alex-Ego/Affective-Computing-VN/tree/master/thesis>
 [LiGame]: <https://github.com/Alex-Ego/Affective-Computing-VN/tree/master/game>
 [LiDesign]: https://github.com/Alex-Ego/Affective-Computing-VN/tree/master/designs
 [PkPG]: <https://www.pygame.org/>
 [PkKe]: <https://keras.io/>
 [PkNLTK]: <https://www.nltk.org/>
 [PkTF]: <https://www.tensorflow.org/>
