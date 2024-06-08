# import libraries 
import argparse
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
import mlflow

# define the main function
def main(args):
    """
    The main function invokes other sub routines in the pipeline
    """
    # enable autolog 
    mlflow.tensorflow.autolog()
    
    # get the data 
    df = get_mlp_data(args.training_data)
    
    # execute the preprocess data function 
    X, Y = preprocess_data(df)
    
    # split the dataset 
    X_train,X_test,Y_train,Y_test = split_data(X, Y)
    
    # train MLP model 
    model, history = train_model(
        X_train,
        X_test,
        Y_train,
        Y_test,
        args.regs,
        args.num_neurons,
        args.num_layers,
        args.epoch,
        args.learning_rate,
        args.batch_size
    )
    
    # evaluate the model 
    evaluate_model(model,X_test,Y_test)
    
    # plot model training graphs
    plot_training(history)

# define a function to read data 
def get_mlp_data(data_path):
    """
    This function reads the data from MLTable format to pandas dataframe
    """
    # use URI file data path to load data to dataframe
    df = pd.read_csv(data_path)
    return df 

# preprocess the data
def preprocess_data(df):
    """ 
    This function applies required preprocessing steps to the data
    """ 
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Fit and transform the 'Area' column
    df['Area'] = label_encoder.fit_transform(df['Area']) 
    
    # Prepare feature matrix X and target vector y
    X = df.drop(columns=['Export_Value'])
    y = df[['Export_Value']]
    
    # Normalize training and testing dataset
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x = scaler_x.fit_transform(X)
    Y = scaler_y.fit_transform(y.values.reshape(-1,1))
    
    return x,Y

# define a function that splits the data 
def split_data(X,Y):
    """ 
    Funtion accepts preprocessed data as input and split for training and testing dataset
    """
    # split dataset for training and testing
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=40)
    
    return x_train, x_test, y_train, y_test

# define architecture for export forecasting model
def train_model(X_train,X_test,Y_train,Y_test,regs,num_neurons,num_layers,epoch,learning_rate,batch_size):
    """
    This function defines the architecture and training of ANN model. 
    Hyper params were used to train model on different set of values. 
    returns: trained model and training history
    """
    # input layer
    inputs = keras.Input(shape=(X_train.shape[1]))
    # l2 regularization for preventing overfitting
    reg = tf.keras.regularizers.l2(l2=regs)
    # first hidden layer
    x = keras.layers.Dense(num_neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg)(inputs)
    # efficiently add more layers
    for _ in range(num_layers - 1):
        x = keras.layers.Dense(num_neurons, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg)(x)
        # Add dropout layer for regularization
        x = keras.layers.Dropout(0.2)(x)
    # output layer
    outputs = keras.layers.Dense(1)(x)
    # set model
    model = keras.Model(inputs=inputs, outputs=outputs, name='export_model')

    # Compile the model for a regression problem
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='mae',
        optimizer=opt,
        metrics=['mae']
    )
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Train the model
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, verbose=1,validation_data=(X_test, Y_test), callbacks=[early_stopping])

    # Save performance
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # print performance
    print('Cost function at epoch 0')
    print('Training MAE=', hist['loss'].values[0])
    print('Validation MAE=', hist['val_loss'].values[0])
    print('Cost function at epoch:', str(epoch))
    print('Training MAE=', hist['loss'].values[-1])
    print('Validation MAE=', hist['val_loss'].values[-1])
    return model, hist

# define a function that evaluates model performance
def evaluate_model(model,X_test,Y_test):
    """
    Evaluate model on Regression evaluation metrics. 
    """
    # predict with model
    y_pred = model.predict(X_test).flatten()
    mean_abs_errors = (mean_absolute_error(Y_test, y_pred))
    root_mean_sqrd_error = np.sqrt(mean_squared_error(Y_test, y_pred))
    r2 = np.round(r2_score(Y_test, y_pred)*100,2)
    
    print("MAE:",mean_abs_errors)
    print("RMSE:",root_mean_sqrd_error)
    print("R2:",r2)

# plot training graphs 
def plot_training(history):
    """ 
    Plot training history (accuracy and loss)
    """
    plt.figure(figsize=(12, 5))
    # Plot Mean Absolute Error (MAE)
    plt.subplot(1, 2, 1)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("Mlp.png")
    mlflow.log_artifact("Mlp.png")

# parse required arguments for model training 
def parse_arguments():
    """
    Function that uses python `argparse` method to define hyperparametre values for model training. 
    """
    # setup the arg parser
    parser = argparse.ArgumentParser()
    
    # add the arguments
    parser.add_argument("--training_data", dest='training_data',type=str)
    parser.add_argument("--regs", dest='regs',type=float, default=0.001)
    parser.add_argument("--learning_rate", dest='learning_rate',type=float, default=0.001)
    parser.add_argument("--epoch", dest='epoch',type=int, default=10)
    parser.add_argument("--batch_size", dest='batch_size',type=int, default=32)
    parser.add_argument("--num_neurons", dest='num_neurons',type=int, default=4)
    parser.add_argument("--num_layers", dest='num_layers',type=int, default=4)
    
    # parse the args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_arguments()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
    
print("Created")
