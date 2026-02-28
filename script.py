import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Импортируем данные
df = pd.read_excel("C:\\Users\\user\Downloads\\breast+cancer+coimbra\\dataR2.xlsx")

# Для удобства будем использовать два параметра:
# Age - возраст пациента
# BMI - индекс массы тела
# Classification - переменная отклика:
# 1 - healthy controls
# 2 - patients

X, y = df[["Age", "BMI"]].values, df["Classification"]

print(df[["Age", "BMI", "Classification"]].head())
print()
print(X.shape, y.shape)
print()

# Формируем обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

print(X_train.shape, y_train.shape)
print()
print(X_test.shape, y_test.shape)

# Применяем Наивный Байесовский классификатор
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print(
    "Количество неправильно классифицированных точек из общего числа %d точек: %d"
    % (X_test.shape[0], (y_test != y_pred).sum())
)
print(y_pred)

# Формируем кросс-валидационную таблицу
dat = {"y_Actual": y_test.to_numpy(), "y_Predicted": y_pred}
dff = pd.DataFrame(dat, columns=["y_Actual", "y_Predicted"])

cross_table = pd.crosstab(
    dff["y_Actual"],
    dff["y_Predicted"],
    rownames=["Actual"],
    colnames=["Predicted"],
    margins=True,
)
print(cross_table)

# Задаем новые точки для прогнозирования
# Замечание. Берем точки из тестовой выборки, а не задаем вручную.
n = X_test[:4]
print()
print("Новые точки для прогнозирования:")
print(n)

# делаем прогноз для новых точек
new_points = gnb.fit(X_train, y_train).predict(n)
print()
print("Прогноз для новых точек:")
print(new_points)

# Визуализируем результат классификации тестовой выборки
plt.figure(1)
c = ["blue" if e == 1 else "pink" for e in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], color=c, linewidths=0.1)
plt.xlabel("Age")
plt.ylabel("BMI")

# Визуализируем новые точки для прогнозирования
plt.figure(2)
plt.scatter(X_test[:, 0], X_test[:, 1], color=c, linewidths=0.1)
plt.scatter(n[0, 0], n[0, 1], color="green", label="Первая точка", linewidths=2)
plt.scatter(n[1, 0], n[1, 1], color="red", label="Вторая точка", linewidths=2)
plt.scatter(n[2, 0], n[2, 1], color="orange", label="Третья точка", linewidths=2)
plt.scatter(n[3, 0], n[3, 1], color="purple", label="Четвертая точка", linewidths=2)
plt.xlabel("Age")
plt.ylabel("BMI")
plt.legend()
plt.show()
