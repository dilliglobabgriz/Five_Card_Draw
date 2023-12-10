# Five_Card_Draw
Five card draw machine learning model

Breakdown of the files in this repository:
- testing.py is the files that includes the hand evalation and comparison functions. To test this out just modify the hand selection at the bottom of the page
- simulation.py creates the training data and saves it to a CSV file for the model to load in.
-  model_training.py takes the CSV created in simulation.py and creates a machine learning model that will predict if a specified card in a five card hand should be swapped.
-  load_model.py is a file that works a demonstration of how the Keras model can be used. It takes a hand of five card, ranks it, organizing it, and gives the best possible card to swap from that hand based on the model (1-5), or (0) if the best decision is to stay
