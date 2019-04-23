#define a regressor using a neural network built in keras and embedded in sklearn
import tensorflow as tf
import keras

#build the model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

def build_NN_reg(drop_rate = 0.2, 
                 max_units = 64, 
                 activation_fct = 'relu', 
                 lr = 0.001, 
                 decay_rate = 1e-5):
    
    model = Sequential()
    
    #xavier initializer for kernel and zeros for bias is default
    model.add(Dense(units = max_units))
    model.add(Activation(activation = activation_fct))
    model.add(Dropout(rate = drop_rate))
    
    model.add(Dense(units = max(round(max_units/4),4)))
    model.add(Activation(activation = activation_fct))
    model.add(Dropout(rate = drop_rate))
    
    model.add(Dense(units = max(round(max_units/16),2)))
    model.add(Activation(activation = activation_fct))
    model.add(Dropout(rate = drop_rate))    
    
    model.add(Dense(1))
    
    adamopt = Adam(lr = lr, decay = decay_rate)
    
    model.compile(optimizer = adamopt, loss = 'mse')
    
    return model


from sklearn.base import BaseEstimator, RegressorMixin
from keras.models import load_model
import inspect
import os


class NNRegressor(BaseEstimator, RegressorMixin): 
    
    def __init__(self, 
                 drop_rate = 0.2, 
                 max_units = 64, 
                 activation_fct = 'relu',
                 lr = 0.001, 
                 decay_rate = 1e-5, 
                 epochs = 10,
                 batch_size = 128,
                 period = 5,
                 validation_split = 0.0,
                 ckpt_path = None,
                 verbose = 1):
        
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        
        #if checkpoint path is given and the file exists load the model
        if self.ckpt_path:
            
            if os.path.isfile(ckpt_path):
                self.reg = load_model(ckpt_path)
            else:
                self.reg = build_NN_reg(
                        drop_rate = self.drop_rate, 
                        max_units = self.max_units, 
                        activation_fct = self.activation_fct, 
                        lr = self.lr, 
                        decay_rate = self.decay_rate)
                
        else:
            self.reg = build_NN_reg(
                    drop_rate = self.drop_rate, 
                    max_units = self.max_units, 
                    activation_fct = self.activation_fct, 
                    lr = self.lr, 
                    decay_rate = self.decay_rate)

        
    def fit(self,X,y):
        
        if self.ckpt_path:
            #define callback function for checkpoints
            #save the best model with the validation loss as a measure
            #thus expects a non-zero validation_split 
            self.ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
                                    self.ckpt_path, 
                                    verbose=self.verbose, 
                                    monitor = 'val_loss',
                                    save_best_only = True,
                                    period = self.period,
                                    mode = 'min')
            #fit the model given a checkpoint path
            self.fit_res_ = self.reg.fit(X,y,
                                    verbose=self.verbose,
                                    epochs = self.epochs,
                                    batch_size = self.batch_size,
                                    validation_split = self.validation_split,
                                    callbacks = [self.ckpt_callback])
        #fit the model ont using any checkpoints 
        #e.g. for cross validation or hyperparameter tuning
        else:
            self.fit_res_ = self.reg.fit(X,y,
                                    verbose=self.verbose,
                                    epochs = self.epochs,
                                    batch_size = self.batch_size,
                                    validation_split = self.validation_split)
    
        return self
        
    
    def predict(self,X,y=None):

        return self.reg.predict(X)