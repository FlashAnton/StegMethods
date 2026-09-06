import numpy as np
from keras.models import Sequential, load_model 
from keras.layers import Dense
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import os


# Функция для генерации данных из файла
def generate_data(file_path):
    """
    Генерирует данные для обучения модели.
    Входные данные: [i-624, i-623, i-227]
    Выходные данные: i
    """
    # Открываем файл с данными
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Преобразуем строки в целые числа
    data = [list(map(int, line.strip().split(', '))) for line in lines]

    # Разделяем данные на входные (X) и выходные (y)
    X = []
    y = []

    for i in range(len(data)):
        if i < 624:
            continue
        input_row = [
            data[i-624][0],  # i-624
            data[i-623][0],  # i-623
            data[i-227][0]   # i-227
        ]
        output_row = data[i][0]  # i
        
        # Преобразуем входные и выходные данные в двоичный вид
        X.append([int(bit) for bit in format(input_row[0], '032b')] +
                 [int(bit) for bit in format(input_row[1], '032b')] +
                 [int(bit) for bit in format(input_row[2], '032b')])
        y.append([int(bit) for bit in format(output_row, '032b')])

    return np.array(X), np.array(y)

# Функция для создания и компиляции модели
def build_model(hp): 
    # Создаем последовательную модель
    model = Sequential()

    # Добавляем скрытый полносвязный слой с 96 нейронами и активацией ReLU
    model.add(Dense(96, input_dim=96, activation='relu'))

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
    file_path = 'mersenne_twist_states.txt'
    X, y = generate_data(file_path)

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train_short = X_train[:600000]
    y_train_short = y_train[:600000]
    
    # Проверка, существует ли уже обученная модель
    if os.path.isfile('twisting_model.h5'):
        # Если модель существует, загружаем ее
        model = load_model('twisting_model.h5')
        model.compile(loss='binary_crossentropy', optimizer=Nadam())  # Компиляция модели после загрузки
    else:
        # Если модели нет, запускаем процесс поиска лучших гиперпараметров с помощью Keras Tuner
        tuner = kt.BayesianOptimization(
            build_model,  # Функция для создания модели
            objective='binary_accuracy',  # Целевая метрика
            max_trials=20,  # Максимальное количество испытаний
            directory='my_dir',  # Директория для сохранения данных тюнинга
            project_name='twisting_model'  # Имя проекта
        )

        # Запуск поиска гиперпараметров
        tuner.search(X_train_short, y_train_short, batch_size=256, epochs=10, validation_data=(X_test, y_test))

        # Получение лучших гиперпараметров после тюнинга
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Создание модели с лучшими гиперпараметрами
        model = tuner.hypermodel.build(best_hps)

    # Обучение модели с найденными или сохраненными гиперпараметрами
    model.fit(X_train, y_train, epochs=10, batch_size=512)

    # Сохранение обученной модели
    model.save('twisting_model.h5')

    # Оценка модели на тестовых данных
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Тестовые потери: {loss}, Тестовая точность: {accuracy}')

# Запуск основной функции
if __name__ == '__main__':
    main()
