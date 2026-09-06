import numpy as np
from keras.models import load_model

def predict_next_number(sequence):
    # Загрузка модели
    model = load_model('C:/PRNG/xorshift/Models/xorshift128_model.h5')

    # Преобразование последовательности в двоичные строки и затем в массивы битов
    binary_sequence = [format(int(num), '032b') for num in sequence]
    bit_arrays = [list(map(int, num)) for num in binary_sequence]

    # Убедитесь, что все списки битов имеют одинаковую длину
    max_len = max(map(len, bit_arrays))
    bit_arrays = [bits + [0]*(max_len-len(bits)) for bits in bit_arrays]

    # Преобразование массивов битов в форму, которую модель может принять
    X = np.array(bit_arrays).reshape(1, -1)

    # Предсказание следующего числа
    y_pred = model.predict(X)

    # Преобразование предсказанных битов обратно в число
    binary_pred = ''.join(map(str, y_pred.round().astype(int)[0]))
    next_number = int(binary_pred, 2)

    return next_number

# Пример использования функции
sequence = [426100976, 3148148308, 2714641884, 1846377973]
print(predict_next_number(sequence))
