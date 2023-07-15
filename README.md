# LSTM-Stock-Price-Prediction-WEB-App-with-Streamlit-Stocksee
**Stock Market Prediction Application  "StockSee"**
This project aims to predict stock market prices using a combination of **LSTM (Long Short-Term Memory)** and **Conv1D (1-D Convolutional Neural Network)** models. The application takes user input in the form of ticker symbol of a stock and presents the performance of the stock in the form of both tables and graphs. The project utilize historical stock market data to train the models and evaluate their performance, the trained model furhter runs for different data inputs through an application.

## Dataset
The dataset used for this project is sourced from <sup>**Yfinance API**</sup>. The dataset contains information about the daily stock prices of a company, including open price, low price, high price, volume, and closing price. The dataset is pre-processed and features such as the 5-day moving average difference (dma) and a binary indicator for positive dma (dma_positive) are added..The dataset is then split into training and testing sets.

## Application
The functionalities provided by the application includ plotting 5-Day trend of any user defined stock, plotting 100 day moving average and 200 day moving average against the close price of any stock, providing a chart that precisely presents the performance accuracy of the LSTM model and also a 10-year data of the mentioned stock. Moreover, with the help of <sup>**Streamlit**</sup> framework an elaborate web application(<sup>**application.py**</sup>) is created for user to interact with where the user can enter a stock ticker and get the analysis of the stock. 

## Dependencies
Make sure you have the following dependencies installed:

Streamlit
YFinance
pandas
numpy
matplotlib
scikit-learn
keras (with TensorFlow backend)
You can install these dependencies using pip:

pip install pandas numpy matplotlib scikit-learn keras tensorflow streamlit yfinance

## Usage
Clone the repository and navigate to the project directory.
Run the **LSTM.ipynb** notebook, this will train model with 'ADOBE' stock data. The notebook/script will load the dataset, preprocess the data, train the models, and generate predictions. The predictions are then compared with the actual stock prices, and the results are displayed in a plot.
execute the **application.py** script.
This will present an elaborate web-page with the neccesary data about ^NSEI(Nifty 50) stock by default, which ofcourse can be change through user input.

## Results
The performance of the models is evaluated using the R-squared (r2) metric, which measures the proportion of the variance in the target variable that is predictable from the input variables. The R-squared values are printed after training the models.

The plot generated after training and testing the models shows the predicted stock prices compared to the actual stock prices for both the training and testing datasets.

Epoch 1/32

110/110 [==============================] - 7s 28ms/step - loss: 0.0015 - val_loss: 6.5159e-04

Epoch 2/32

110/110 [==============================] - 2s 21ms/step - loss: 5.3944e-04 - val_loss: 4.9965e-04

Epoch 3/32

110/110 [==============================] - 2s 21ms/step - loss: 4.7850e-04 - val_loss: 8.0414e-04

Epoch 4/32

110/110 [==============================] - 2s 21ms/step - loss: 5.2687e-04 - val_loss: 7.0148e-04

Epoch 5/32

110/110 [==============================] - 2s 21ms/step - loss: 4.0620e-04 - val_loss: 8.3856e-04

Epoch 6/32

110/110 [==============================] - 3s 28ms/step - loss: 3.4310e-04 - val_loss: 3.5203e-04

Epoch 7/32

110/110 [==============================] - 3s 23ms/step - loss: 3.5966e-04 - val_loss: 3.8621e-04

Epoch 8/32

110/110 [==============================] - 3s 23ms/step - loss: 3.2312e-04 - val_loss: 3.5566e-04

Epoch 9/32

110/110 [==============================] - 3s 26ms/step - loss: 3.3792e-04 - val_loss: 5.8536e-04

Epoch 10/32

110/110 [==============================] - 3s 24ms/step - loss: 3.0353e-04 - val_loss: 4.7943e-04

Epoch 11/32

110/110 [==============================] - 3s 23ms/step - loss: 3.2953e-04 - val_loss: 4.0473e-04

Epoch 12/32

110/110 [==============================] - 3s 24ms/step - loss: 3.1095e-04 - val_loss: 6.9332e-04

Epoch 13/32

110/110 [==============================] - 3s 24ms/step - loss: 3.0571e-04 - val_loss: 4.0930e-04

Epoch 14/32

110/110 [==============================] - 3s 23ms/step - loss: 3.0445e-04 - val_loss: 6.5094e-04

Epoch 15/32

110/110 [==============================] - 3s 24ms/step - loss: 2.9691e-04 - val_loss: 4.0356e-04

Epoch 16/32

110/110 [==============================] - 3s 26ms/step - loss: 2.8319e-04 - val_loss: 2.8028e-04

Epoch 17/32

110/110 [==============================] - 3s 24ms/step - loss: 2.5248e-04 - val_loss: 0.0011

Epoch 18/32

110/110 [==============================] - 2s 23ms/step - loss: 3.2498e-04 - val_loss: 3.0697e-04

Epoch 19/32

110/110 [==============================] - 2s 22ms/step - loss: 2.7388e-04 - val_loss: 5.3142e-04

Epoch 20/32

110/110 [==============================] - 2s 23ms/step - loss: 2.9956e-04 - val_loss: 7.0261e-04

Epoch 21/32

110/110 [==============================] - 2s 22ms/step - loss: 2.9840e-04 - val_loss: 3.8025e-04

Epoch 22/32

110/110 [==============================] - 3s 23ms/step - loss: 3.7012e-04 - val_loss: 3.2637e-04

Epoch 23/32

110/110 [==============================] - 2s 22ms/step - loss: 3.1930e-04 - val_loss: 2.5594e-04

Epoch 24/32

110/110 [==============================] - 3s 25ms/step - loss: 3.0045e-04 - val_loss: 6.1684e-04

Epoch 25/32

110/110 [==============================] - 3s 25ms/step - loss: 2.7133e-04 - val_loss: 2.5089e-04

Epoch 26/32

110/110 [==============================] - 3s 26ms/step - loss: 2.8115e-04 - val_loss: 2.3883e-04

Epoch 27/32

110/110 [==============================] - 3s 24ms/step - loss: 2.6196e-04 - val_loss: 3.4572e-04

Epoch 28/32

110/110 [==============================] - 2s 22ms/step - loss: 2.5604e-04 - val_loss: 2.4187e-04

Epoch 29/32

110/110 [==============================] - 2s 22ms/step - loss: 2.5818e-04 - val_loss: 3.3302e-04

Epoch 30/32

110/110 [==============================] - 3s 30ms/step - loss: 3.4629e-04 - val_loss: 2.3294e-04

Epoch 31/32

110/110 [==============================] - 2s 22ms/step - loss: 2.7871e-04 - val_loss: 7.3923e-04

Epoch 32/32

110/110 [==============================] - 2s 21ms/step - loss: 3.0125e-04 - val_loss: 2.4395e-04

61/61 [==============================] - 1s 10ms/step

16/16 [==============================] - 0s 10ms/step

r2 train:  0.9927794860128337

r2 test:  0.967680286736371

The models were trained for 32 epochs with a batch size of 16. The training loss and validation loss for each epoch are as follows:

Epoch 1/32 - loss: 0.0015 - val_loss: 6.5159e-04

Epoch 2/32 - loss: 5.3944e-04 - val_loss: 4.9965e-04

...

Epoch 31/32 - loss: 2.7871e-04 - val_loss: 7.3923e-04

Epoch 32/32 - loss: 3.0125e-04 - val_loss: 2.4395e-04

After training, the models were evaluated using the R-squared (r2) metric. The R-squared values obtained are as follows:

* Training set: 0.9927794860128337
* Testing set: 0.967680286736371
* 
The R-squared value indicates the goodness of fit of the models. A value closer to 1 indicates a better fit to the data.

The plot generated after training and testing the models shows the predicted stock prices compared to the actual stock prices for both the training and testing datasets.

These results demonstrate the effectiveness of the **LSTM **and** Conv1D** models in predicting stock market prices based on the provided dataset.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.
