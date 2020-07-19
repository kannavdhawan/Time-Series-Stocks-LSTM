
- Instructions to run: 
    - For Training:
        - `python3 Time_series_stocks_LSTM/train_RNN.py `
    - For Testing:
        - `python3 Time_series_stocks_LSTM/test_RNN.py `
- Note: 
    - Models Run on Batch size=10 and epochs 100 unless implitly mentioned.
    - For Detailed Model selection, please refer the report added in pdf format.
### Epochs= 100 | Batch size= 50 | model=LSTM_RNN(add_dense_32=False,add_dense_20=False,add_dense_10=False,opt='adam')
- Train RMSE : 4.883128665710501
- Train MSE :  23.84494556588362
- Train MAE :  3.3057946562088945
- Test RMSE : 5.183044062310987
- Test MSE :  26.863945751857184
- Test MAE :  3.28921729373679
### Epochs= 50 | Batch size= 10 | model=LSTM_RNN(add_dense_32=False,add_dense_20=False,add_dense_10=False,opt='adam')
- Train RMSE : 3.9753924784099643
- Train MSE :  15.803745357398519
- Train MAE :  2.5186327347305175
- Test RMSE : 4.0408443398364495
- Test MSE :  16.328422978788275
- Test MAE :  2.3988517336250923

#### Note:
1. Batch_size=10
2. Epochs=100

### Model | Epochs: 100, Batch_size: 10 
Layer (type)        |         Output Shape   |           Param #   
----|-----|-----
lstm (LSTM)          |        (None, 64)      |          17664     
dense (Dense)         |       (None, 1)        |         65        

- Total params: 17,729
- Trainable params: 17,729
- Non-trainable params: 0
- Loss and MAE on Train set:  [0.00010530659346841276, 0.006131733302026987]
- Random Testing for Training data 

    - Predicted:  [110.774376]
    - Target:  [112.30999756]

- Different Losses after inverting the prices to real scale for Train Data: 

    - Train RMSE : 2.942291786509576
    - Train MSE :  8.657080956961714
    - Train MAE :  1.7580904391031613
- Loss and MAE on Test set:  [0.00010609572200337425, 0.005971579812467098]
- Random Testing for Test data 
    - Predicted:  [219.45885]
    - Target:  [222.30000305]
- Different Losses after inverting the prices to real scale for Test Data: 
    - Test RMSE : 2.9532958992995972
    - Test MSE :  8.721956668819818
    - Test MAE :  1.7121710941709316
### model=LSTM_RNN(add_dense_32=False,add_dense_20=False,add_dense_10=False,opt='adam')

Layer (type)         |        Output Shape        |      Param #   
---|---|---
lstm (LSTM)          |       (None, 3, 32)         |    4736      
lstm_1 (LSTM)         |       (None, 10)            |    1720      
dense (Dense)          |      (None, 1)              |   11        

- Total params: 6,467
- Trainable params: 6,467
- Non-trainable params: 0

- Loss and MAE on Train set:  [0.0001275644899578765, 0.006971674505621195]
- Random Testing for Training data 
    - Predicted:  [110.534256]
    - Target:  [112.30999756]
- Different Losses after inverting the prices to real scale for Train Data: 
    - Train RMSE : 3.2383429500285756
    - Train MSE :  10.486865061999778
    - Train MAE :  1.9989185962525102

- Loss and MAE on Test set:  [0.00011769217962864786, 0.006533808074891567]
- Random Testing for Test data 
    - Predicted:  [219.60915]
    - Target:  [222.30000305]


- Different Losses after inverting the prices to real scale for Test Data: 

    - Test RMSE : 3.110511633396325
    - Test MSE :  9.675282621493873
    - Test MAE :  1.8733735628406294

_________________________________________________________________
### model=LSTM_RNN(add_dense_32=False,add_dense_20=False,add_dense_10=False,opt='sgd')

Layer (type)         |        Output Shape        |      Param #   
---|---|---
lstm (LSTM)          |       (None, 3, 32)         |    4736      
lstm_1 (LSTM)         |       (None, 10)            |    1720      
dense (Dense)          |      (None, 1)              |   11        

- Total params: 6,467
- Trainable params: 6,467
- Non-trainable params: 0

- Loss and MAE on Train set:  [0.0003418474516365677, 0.012409715913236141]

- Random Testing for Training data 
    - Predicted:  [108.89464]
    - Target:  [112.30999756]

- Different Losses after inverting the prices to real scale for Train Data: 

    - Train RMSE : 5.301199683486834
    - Train MSE :  28.10271808420091
    - Train MAE :  3.558113618922315


- Loss and MAE on Test set:  [0.00029108498711138964, 0.011396192945539951]
- Random Testing for Test data 
    - Predicted:  [224.25378]
    - Target:  [222.30000305]
- Different Losses after inverting the prices to real scale for Test Data: 
    - Test RMSE : 4.891790284082781
    - Test MSE :  23.929612183446697
    - Test MAE :  3.2675161386990736

_________________________________________________________________
### model=LSTM_RNN(add_dense_32=True,add_dense_20=False,add_dense_10=False,opt='adam')

Layer (type)            |     Output Shape   |           Param #   
-----|-----|--------
lstm (LSTM)           |       (None, 3, 32)  |           4736      
lstm_1 (LSTM)           |     (None, 10)   |             1720      
dense (Dense)         |       (None, 32)      |          352       
dense_1 (Dense)        |      (None, 1)    |             33        

- Total params: 6,841
- Trainable params: 6,841
- Non-trainable params: 0

- Loss and MAE on Train set:  [0.00016918503388296813, 0.009508706629276276]
- Random Testing for Training data 

    - Predicted:  [112.50755]
    - Target:  [112.30999756]

- Different Losses after inverting the prices to real scale for Train Data: 

    - Train RMSE : 3.729400137310899
    - Train MSE :  13.908425384174555
    - Train MAE :  2.726336473762242
- Loss and MAE on Test set:  [0.00018008229380939156, 0.009327458217740059]
- Random Testing for Test data 
    - Predicted:  [222.41693]
    - Target:  [222.30000305]


- Different Losses after inverting the prices to real scale for Test Data: 

    - Test RMSE : 3.847631582467061
    - Test MSE :  14.804268794397979
    - Test MAE :  2.6743684561246277
_________________________________________________________________
### model=LSTM_RNN(add_dense_32=True,add_dense_20=True,add_dense_10=False,opt='adam')

Layer (type)         |        Output Shape |              Param #   
----|----|----
lstm (LSTM) |                 (None, 3, 32)  |           4736      
lstm_1 (LSTM)|                (None, 10)      |          1720      
dense (Dense) |               (None, 32)       |         352       
dense_1 (Dense)|              (None, 20)        |        660       
dense_2 (Dense) |             (None, 1)          |       21        

- Total params: 7,489
- Trainable params: 7,489
- Non-trainable params: 0
- Loss and MAE on Train set:  [0.00012914663238916546, 0.0070859952829778194]
- Random Testing for Training data 
    - Predicted:  [110.60154]
    - Target:  [112.30999756]
- Different Losses after inverting the prices to real scale for Train Data: 
    - Train RMSE : 3.258363033394074
    - Train MSE :  10.616929657389031
    - Train MAE :  2.031696349958782
- Loss and MAE on Test set:  [0.0001382699701935053, 0.006779192015528679]
- Random Testing for Test data 
    - Predicted:  [221.49329]
    - Target:  [222.30000305]
- Different Losses after inverting the prices to real scale for Test Data: 

    - Test RMSE : 3.3714902330506322
    - Test MSE :  11.366946391555805
    - Test MAE :  1.94372953217605

_________________________________________________________________
### model=LSTM_RNN(add_dense_32=True,add_dense_20=True,add_dense_10=True,opt='adam')

Layer (type)   |              Output Shape  |            Param #   
-----|-----|------
lstm (LSTM)       |           (None, 3, 32)   |          4736      
lstm_1 (LSTM)      |          (None, 10)       |         1720      
dense (Dense)       |         (None, 32)        |        352       
dense_1 (Dense)      |        (None, 20)         |       660       
dense_2 (Dense)       |       (None, 10)          |      210       
dense_3 (Dense)        |      (None, 1)            |     11        

- Total params: 7,689
- Trainable params: 7,689
- Non-trainable params: 0
- Loss and MAE on Train set:  [0.0001241696154465899, 0.006924543529748917]
- Random Testing for Training data 

    - Predicted:  [109.86439]
    - Target:  [112.30999756]


- Different Losses after inverting the prices to real scale for Train Data: 

    - Train RMSE : 3.194961908026734
    - Train MSE :  10.207781593741828
    - Train MAE :  1.9854046557950484
- Loss and MAE on Test set:  [0.00013497361214831471, 0.0068069701083004475]
- Random Testing for Test data 
    - Predicted:  [221.05104]
    - Target:  [222.30000305]

- Different Losses after inverting the prices to real scale for Test Data: 

    - Test RMSE : 3.3310599438666264
    - Test MSE :  11.095960349632731
    - Test MAE :  1.9516947819636419
