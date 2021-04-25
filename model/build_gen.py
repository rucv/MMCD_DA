import svhn2mnist
import usps
import syn2gtrsb
import mnist2mnistm
#import syndig2svhn

'''
    1. usps -> mnist
    2. mnist -> usps
    3. svhn -> mnist
    4. mnist -> mnist_m
    5. syn -> mnist
'''
def Generator(source, target, pixelda=False):
    if source == 'svhn' or target == 'svhn':
        return svhn2mnist.Feature()
    elif source == 'usps' or target == 'usps':
        return usps.Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()
    elif target == 'mnist_m':
        return mnist2mnistm.Feature()

def Classifier(source, target):
    if source == 'svhn'or target == 'svhn':
        return svhn2mnist.Predictor()
    elif source == 'usps' or target == 'usps':
        return usps.Predictor()
    elif source == 'synth':
        return syn2gtrsb.Predictor()
    elif target == 'mnist_m':
        return mnist2mnistm.Predictor()

def Discriminator(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Domain_Classifier()
    elif source == 'svhn':
        return svhn2mnist.Domain_Classifier()
    elif source == 'synth':
        return syn2gtrsb.Domain_Classifier()
    elif target == 'mnist_m':
        return mnist2mnistm.Domain_Classifier()
