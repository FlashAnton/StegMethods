import numpy as np

# Константы для 32-битной реализации
W, N, M, R = 32, 624, 397, 31
A = 0x9908B0DF
U, D = 11, 0xFFFFFFFF
S, B = 7, 0x9D2C5680
T, C = 15, 0xEFC60000
L = 18
F = 1812433253

MT = [0] * N  # Массив для хранения состояния генератора
index = N + 1  # Индекс для отслеживания текущего состояния генератора

# Маски для операций с битами
MASK = (1 << W) - 1  # Маска для обрезания значений до 32 бит
LOWER_MASK = (1 << R) - 1  # Нижняя маска для отделения младших битов
UPPER_MASK = MASK & ~LOWER_MASK  # Верхняя маска для отделения старших битов

# Функция инициализации генератора с заданным значением seed
def seed_mt(seed):
    global index, MT
    index = N  # Сброс индекса
    MT[0] = seed  # Установка начального значения в массив состояния
    for i in range(1, N):
        MT[i] = MASK & (F * (MT[i - 1] ^ (MT[i - 1] >> (W - 2))) + i)

# Функция для извлечения случайного числа из состояния
def extract_number():
    global index
    # Проверка необходимости обновления состояния
    if index >= N:
        if index > N:
            seed_mt(5489)  # Использование стандартного значения для инициализации
        twist()  # Обновление состояния генератора

    y = MT[index]
    y = xorshifter(y)  # Применение операции xorshift

    index += 1
    return MASK & y

# Функция, реализующая операцию xorshift для числа y
def xorshifter(y):
    y = y ^ ((y >> U) & D)
    y = y ^ ((y << S) & B)
    y = y ^ ((y << T) & C)
    y = y ^ (y >> L)
    return y

# Функция для извлечения текущего состояния генератора
def extract_state():
    global index
    # Проверка необходимости обновления состояния
    if index >= N:
        if index > N:
            seed_mt(5489)
        twist()

    y = MT[index]
    index += 1
    return MASK & y

# Функция для обновления состояния генератора (twist)
def twist():
    global MT, index
    for i in range(N):
        # Определение значений старших и младших битов для генерации нового состояния
        x = (MT[i] & UPPER_MASK) + (MT[(i + 1) % N] & LOWER_MASK)
        xA = x >> 1
        if x % 2 != 0:  # Проверка, является ли младший бит равным 1
            xA = xA ^ A
        MT[i] = MT[(i + M) % N] ^ xA  # Обновление значения в массиве состояния
    index = 0  # Сброс индекса после обновления

# Основная функция программы
def main():
    n = 10000  # Количество случайных чисел для генерации
    seed_mt(9568989)  # Инициализация генератора начальным значением (seed)
    vxorshifter = np.vectorize(xorshifter)  # Векторизация функции xorshifter
    states = np.array([extract_state() for _ in range(n)])  # Генерация начальных состояний
    mt_randoms = vxorshifter(states)  # Применение xorshift к состояниям

    # Сохранение состояний генератора в файл
    with open('C:/PRNG/MT/mersenne_twist_states.txt', 'w') as f:
        for item in states:
            f.write("%s\n" % item)

    # Сохранение результатов xorshift и состояний генератора в файл
    with open('C:/PRNG/MT/mersenne_twist_xorshifter.txt', 'w') as f:
        for i in range(n):
            f.write("%s, %s\n" % (mt_randoms[i], states[i]))

    # Сохранение окончательных результатов в файл
    with open('C:/PRNG/MT/mt_final.txt', 'w') as f:
        for number in mt_randoms:
            f.write("%s\n" % number)

if __name__ == '__main__':
    main()
