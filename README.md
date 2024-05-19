# LSTM-Stock-Price-Prediction-WEB-App-with-Streamlit-Stocksee
# Try the finished web app here- (https://stocksee.streamlit.app/)

**Stock Market Prediction Application  "StockSee"**
This project aims to predict stock market prices using a combination of **LSTM (Long Short-Term Memory)** and **Conv1D (1-D Convolutional Neural Network)** models. The application takes user input in the form of ticker symbol of a stock and presents the performance of the stock in the form of both tables and graphs. The project utilize historical stock market data to train the models and evaluate their performance, the trained model furhter runs for different data inputs through an application.

<img width="571" alt="Screenshot 2023-07-17 005029" src="https://github.com/uditsathe/LSTM-Stock-Price-Prediction-WEB-App-with-Streamlit-Stocksee/assets/102481732/400ec0ed-c05a-40a1-a694-b151bbd10d38">

## Dataset
The dataset used for this project is sourced from <sup>**Yfinance API**</sup>. The dataset contains information about the daily stock prices of a company, including open price, low price, high price, volume, and closing price. The dataset is pre-processed and features such as the 5-day moving average difference (dma) and a binary indicator for positive dma (dma_positive) are added..The dataset is then split into training and testing sets.

## Application
The functionalities provided by the application includ plotting 5-Day trend of any user defined stock, plotting 100 day moving average and 200 day moving average against the close price of any stock, providing a chart that precisely presents the performance accuracy of the LSTM model and also a 10-year data of the mentioned stock. Moreover, with the help of <sup>**Streamlit**</sup> framework an elaborate web application(<sup>**application.py**</sup>) is created for user to interact with where the user can enter a stock ticker and get the analysis of the stock. 

<img width="582" alt="Screenshot 2023-07-17 005101" src="https://github.com/uditsathe/LSTM-Stock-Price-Prediction-WEB-App-with-Streamlit-Stocksee/assets/102481732/248d647d-8d04-4446-9394-17461492ff43">
<img width="581" alt="Screenshot 2023-07-17 005141" src="https://github.com/uditsathe/LSTM-Stock-Price-Prediction-WEB-App-with-Streamlit-Stocksee/assets/102481732/091029f0-5778-4e86-acb3-b1d193e6ce9d">
<img width="629" alt="Screenshot 2023-07-17 005216" src="https://github.com/uditsathe/LSTM-Stock-Price-Prediction-WEB-App-with-Streamlit-Stocksee/assets/102481732/c4983c4a-1d78-4de8-a074-7cc1f6a47b13">


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

........

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

The R-squared value indicates the goodness of fit of the models. A value closer to 1 indicates a better fit to the data.

The plot generated after training and testing the models shows the predicted stock prices compared to the actual stock prices for both the training and testing datasets.

These results demonstrate the effectiveness of the **LSTM **and** Conv1D** models in predicting stock market prices based on the provided dataset.

<img width="573" alt="Screenshot 2023-07-17 005234" src="https://github.com/uditsathe/LSTM-Stock-Price-Prediction-WEB-App-with-Streamlit-Stocksee/assets/102481732/9e5b7ce1-bc79-4fbf-bc9d-f9c889e1030b">


## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.
