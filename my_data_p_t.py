# inporting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importin the database
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # копируем все кроме последнего столбца
y = dataset.iloc[:, 3].values # копируем только последний столбец

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer( # ищем пустые поля и заполняем из срендними значениями
  missing_values = 'NaN',
  strategy = 'mean',
  axis = 0
)
imputer = imputer.fit(X[:, 1:3]) #? прокидываем 2,3 и 4 столбцы
X[:, 1:3] = imputer.transform(X[:, 1:3]) #? вставляем данные в X

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lavelencoder_X = LabelEncoder()
X[:, 0] = lavelencoder_X.fit_transform(X[:, 0]) # пререводим страны в цифры
onehotencoder = OneHotEncoder(categorical_features = [0]) # настраиваем переводчик
X = onehotencoder.fit_transform(X).toarray() # по первому столбцу делаем разные катеогрии
# т.е. было [1, 2, 3] а будет три стобца [[1,00],[010],[001]]

lavelencoder_y2 = LabelEncoder()
y = lavelencoder_y2.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size = 0.2, # 0.25 0.3
  random_state = 0 # 42
)
# получается в train - данные по которым мы будем обучать нейронку
# test - данные по которым мы будем проверять нейронку на работоспособность
# test_size = 0.2 - соотношение 0.8(80%) - обучение и 0.2(20%) - тестирование


# Feature Scaling приводим все цифры к интервалу -1 до 1, смотря какой способ
# так быстрее будет работать обучение и не будет перевеса у данных разных
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)





60*8 + 120*2 + 90*8 +30*8









