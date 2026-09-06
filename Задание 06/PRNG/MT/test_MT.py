import numpy as np
from keras.models import load_model


# Constants for 32-bit implementation
U, D = 11, 0xFFFFFFFF
S, B = 7, 0x9D2C5680
T, C = 15, 0xEFC60000
L = 18


def xorshifter(y):
    y = y ^ ((y >> U) & D)
    y = y ^ ((y << S) & B)
    y = y ^ ((y << T) & C)
    y = y ^ (y >> L)
    return y

def preprocess_input(number):
    """
    Преобразует входное число в бинарный вид и форматирует его как массив.
    """
    # Преобразуем число в 32-битное бинарное представление
    binary_string = format(number, '032b')
    
    # Преобразуем бинарную строку в массив значений 0 и 1
    binary_array = np.array([int(bit) for bit in binary_string])
    
    # Изменяем форму массива для соответствия входному размеру модели (1, 32)
    return binary_array.reshape(1, 32)

def predict_with_model(model, input_number):
    """
    Использует загруженную модель для предсказания на основе входного числа.
    """
    # Преобразуем входное число в подходящий формат
    processed_input = preprocess_input(input_number)
    
    # Выполняем предсказание
    prediction = model.predict(processed_input)
    
    # Преобразуем предсказание в бинарный вид
    predicted_binary = (prediction > 0.5).astype(int)
    
    # Преобразуем бинарное значение обратно в целое число
    predicted_number = int(''.join(predicted_binary[0].astype(str)), 2)
    
    return predicted_number

def main():
    # Загрузка моделей
    model_tempering = load_model('C:/PRNG/MT/Models/MT_tempering_model.h5')
    model_twisting = load_model('C:/PRNG/MT/Models/MT_twisting_model.h5')

    # Заданные числа
    numbers = [3992670690, 3823185381, 721782752]
    
    # Получаем результаты предсказания для каждого числа
    results = [predict_with_model(model_tempering, number) for number in numbers]
    
    # Объединяем результаты в одну последовательность бит
    combined_bits = np.concatenate([preprocess_input(result)[0] for result in results])
    
    # Обрабатываем объединённую последовательность битов через модель twisting
    combined_bits = combined_bits.reshape(1, 96)  # Изменяем форму для входа модели twisting
    twisting_prediction = model_twisting.predict(combined_bits)
    
    # Преобразуем предсказание в бинарный вид
    twisting_predicted_binary = (twisting_prediction > 0.5).astype(int)
    
    # Преобразуем бинарное значение обратно в целое число
    twisting_predicted_number = int(''.join(twisting_predicted_binary[0].astype(str)), 2)
    final = xorshifter(twisting_predicted_number)
    # Вывод результатов
    for i, result in enumerate(results):
        print(f"Результат предсказания для числа {numbers[i]}: {result}")
    
    print(f"Результат после применения twisting: {twisting_predicted_number}")
    print(f"Следующее число в последовательности: {final}")
if __name__ == '__main__':
    main()
