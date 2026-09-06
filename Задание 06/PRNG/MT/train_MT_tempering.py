import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Nadam, Adam
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import os

# Функция для генерации данных из файла
def generate_data():
    # Открываем файл с данными
    with open('mersenne_twist_xorshifter.txt', 'r') as f:
        lines = f.readlines()

    # Преобразуем строки в целые числа
    data = [list(map(int, line.strip().split(', '))) for line in lines]

    # Преобразуем данные в массивы X и y
    X = np.array([x[0] for x in data])
    y = np.array([x[1] for x in data])

    # Преобразуем числа в двоичные строки
    X = np.array([list(map(int, format(x, '032b'))) for x in X])
    y = np.array([list(map(int, format(x, '032b'))) for x in y])

    return X, y

# Функция для создания и компиляции модели
def build_model(hp): 
    # Создаем последовательную модель
    model = Sequential()

    # Добавляем первый полносвязный слой с 640 нейронами и активацией ReLU
    model.add(Dense(640, input_dim=32, activation='relu'))

    # Добавляем выходной слой с 32 нейронами и сигмоидной активацией
    model.add(Dense(32, activation='sigmoid'))

    # Настраиваем оптимизатор Nadam с гиперпараметрами, которые будут настроены с помощью Keras Tuner
    opt = Nadam(
        learning_rate=hp.Float("learning_rate", 10**(-5), 10**(-3), sampling="log"),
        epsilon=hp.Float("epsilon", 1e-7, 1e-5, sampling="log"),
        beta_1=hp.Float("beta_1", .8, .9, sampling="reverse_log"),
        beta_2=hp.Float("beta_2", .8, .9, sampling="reverse_log"),
    )

    # Компилируем модель с функцией потерь 'binary_crossentropy' и метрикой 'binary_accuracy'
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    
    return model

# Основная функция для выполнения всех шагов
def main():
    # Генерация данных
    X, y = generate_data()

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train_short = X_train[:600000]
    y_train_short = y_train[:600000]
    
    # Проверка, существует ли уже обученная модель
    if os.path.isfile('MT_tempering_model.h5'):
        # Если модель существует, загружаем ее
        model = load_model('MT_tempering_model.h5')
        model.compile(loss='binary_crossentropy', optimizer=Adam())  # Компиляция модели после загрузки
    else:
        # Если модели нет, запускаем процесс поиска лучших гиперпараметров с помощью Keras Tuner
        tuner = kt.BayesianOptimization(
            build_model,  # Функция для создания модели
            objective='binary_accuracy',  # Целевая метрика
            max_trials=20,  # Максимальное количество испытаний
            directory='my_dir',  # Директория для сохранения данных тюнинга
            project_name='MT_tempering'  # Имя проекта
        )

        # Запуск поиска гиперпараметров
        tuner.search(X_train_short, y_train_short, batch_size=256, epochs=10, validation_data=(X_test, y_test))

        # Получение лучших гиперпараметров после тюнинга
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Создание модели с лучшими гиперпараметрами
        model = tuner.hypermodel.build(best_hps)

    # Обучение модели с найденными или сохраненными гиперпараметрами
    model.fit(X_train, y_train, epochs=5, batch_size=512)

    # Сохранение обученной модели
    model.save('MT_tempering_model.h5')

    # Оценка модели на тестовых данных
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Тестовые потери: {loss}, Тестовая точность: {accuracy}')

# Запуск основной функции
if __name__ == '__main__':
    main()
