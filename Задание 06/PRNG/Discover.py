import numpy as np
from keras.models import load_model
import tempfile
import os
import subprocess
import tensorflow as tf
import logging
import warnings

# Отключение вывода TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключение всех сообщений TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Отключение предупреждений от TensorFlow и других библиотек
logging.getLogger('absl').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Список для накопления сообщений
output_messages = []

# Функция для прогнозирования следующего числа с использованием модели xorshift128
def predict_next_number_xorshift(sequence):
    model = load_model('C:/PRNG/xorshift/Models/xorshift128_model.h5')
    binary_sequence = [format(int(num), '032b') for num in sequence]
    bit_arrays = [list(map(int, num)) for num in binary_sequence]
    X = np.array(bit_arrays).reshape(1, 128)
    y_pred = model.predict(X)
    binary_pred = ''.join(map(str, y_pred.round().astype(int)[0]))
    next_number = int(binary_pred, 2)
    return next_number

# Функция для обработки и проверки чисел Mersenne Twister
def process_mersenne_twister(sequence):
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
        binary_string = format(number, '032b')
        binary_array = np.array([int(bit) for bit in binary_string])
        return binary_array.reshape(1, 32)

    def predict_with_model(model, input_number):
        processed_input = preprocess_input(input_number)
        prediction = model.predict(processed_input)
        predicted_binary = (prediction > 0.5).astype(int)
        predicted_number = int(''.join(predicted_binary[0].astype(str)), 2)
        return predicted_number

    model_tempering = load_model('C:/PRNG/MT/Models/MT_tempering_model.h5')
    model_twisting = load_model('C:/PRNG/MT/Models/MT_twisting_model.h5')

    if len(sequence) >= 625:  # Убедитесь, что есть хотя бы 625 чисел
        # Выбираем 1-е, 2-е и 398-е числа
        numbers = [sequence[0], sequence[1], sequence[397]]

        tempered_results = [predict_with_model(model_tempering, number) for number in numbers]

        # Объединяем их в одну последовательность из 96 бит
        combined_bits = ''.join(format(result, '032b') for result in tempered_results)
        combined_bits_array = np.array([int(bit) for bit in combined_bits]).reshape(1, 96)

        # Прогоняем через модель twisting
        twisting_prediction = model_twisting.predict(combined_bits_array)
        twisting_predicted_binary = (twisting_prediction > 0.5).astype(int)
        twisting_predicted_number = int(''.join(twisting_predicted_binary[0].astype(str)), 2)
     
        # Применяем функцию xorshifter
        final_number = xorshifter(twisting_predicted_number)
     
        return final_number
    else:
        raise ValueError("Недостаточно чисел для проверки Mersenne Twister.")

# Главная функция
def main():
    global output_messages
    
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_file:
        temp_file_name = temp_file.name
        temp_file.write("Введите числа, по одному на строку. Сохраните и закройте файл для продолжения...\n")
    
    subprocess.run(['notepad', temp_file_name], check=True)

    with open(temp_file_name, 'r') as file:
        next(file)
        sequence = [int(line.strip()) for line in file if line.strip().isdigit()]
    
    os.remove(temp_file_name)

    if len(sequence) >= 4:
        predicted_number = predict_next_number_xorshift(sequence[:4])
        if predicted_number == sequence[4]:
            output_messages.append("Последовательность соответствует xorshift128.")
            next_numbers = []
            for i in range(10):
                sequence.append(predict_next_number_xorshift(sequence[-4:]))
                next_numbers.append(sequence[-1])
            output_messages.append(f"Следующие 10 чисел для xorshift128: {next_numbers}")
        else:
            output_messages.append("Последовательность не соответствует xorshift128.")
    else:
        output_messages.append("Недостаточно чисел для проверки xorshift128.")

    if len(sequence) >= 625:
        try:
            predicted_number_mt = process_mersenne_twister(sequence)
            if predicted_number_mt == sequence[624]:
                output_messages.append("Последовательность соответствует Mersenne Twister.")
                next_numbers_mt = []
                for i in range(10):
                    predicted_number_mt = process_mersenne_twister(sequence[-625:])
                    next_numbers_mt.append(predicted_number_mt)
                    sequence.append(predicted_number_mt)
                output_messages.append(f"Следующие 10 чисел для Mersenne Twister: {next_numbers_mt}")
            else:
                output_messages.append("Последовательность не соответствует Mersenne Twister.")
        except ValueError as e:
            output_messages.append(f"Ошибка при проверке Mersenne Twister: {e}")
    else:
        output_messages.append("Недостаточно чисел для проверки Mersenne Twister.")

    # Вывод всех сообщений в конце
    for message in output_messages:
        print(message)

if __name__ == '__main__':
    main()
