import cnn_model as cnn
import model as m
def predict(input):
    config = {'hidden_size': 512} # same config 
    model = m.Classifier(cnn.ConvolutionalModel, cnn.Config(name_suffix='THE_BEAST', **config))
    pred = model.predict(input)
    return pred
