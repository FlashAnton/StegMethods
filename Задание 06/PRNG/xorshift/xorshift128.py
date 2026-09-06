def xorshift128():
    x = 352454534
    y = 6543265
    z = 23567864
    w = 65432

    def _random():
        nonlocal x, y, z, w
        t = x ^ ((x << 11) & 0xFFFFFFFF)  # 32-bit
        x, y, z = y, z, w
        w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
        return w

    return _random


def main():
    with open('xorshift128.txt', 'w') as f:
        r = xorshift128()
        for _ in range(3000000):
            f.write(str(r()) + '\n')


if __name__ == '__main__':
    main()
