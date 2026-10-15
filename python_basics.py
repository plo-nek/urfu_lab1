"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiou")
    return sum(1 for ch in text.lower() if ch in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    total = 0
    for p, e in zip(predicted, expected):
        total += (p - e) ** 2
    return total / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    parts = []
    divisor = 2
    n = number
    while divisor * divisor <= n:
        if n % divisor == 0:
            power = 0
            while n % divisor == 0:
                n //= divisor
                power += 1
            if power == 1:
                parts.append(f"({divisor})")
            else:
                parts.append(f"({divisor}**{power})")
        divisor += 1
    if n > 1:
        parts.append(f"({n})")
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(d) for d in str(number)]
    length = len(digits)

    if length % 2 == 0:
        # две средние цифры исключаются
        mid = length // 2
        left = digits[:mid - 1]
        right = digits[mid + 1:]
    else:
        mid = length // 2
        left = digits[:mid]
        right = digits[mid + 1:]

    return sum(left) == sum(right)

# 1
print (count_vowels(TextInput("Hello World")) == 3)
print (count_vowels(TextInput("")) == 0)
print (count_vowels(TextInput("XYZ")) == 0)

# 2
print (has_unique_characters(TextInput("abc")) is True)
print (has_unique_characters(TextInput("aabc")) is False)
print (has_unique_characters(TextInput("")) is True)

# 3
print (count_one_bits(PositiveIntegerInput(7)) == 3)
print (count_one_bits(PositiveIntegerInput(1024)) == 1)
print (count_one_bits(PositiveIntegerInput(1)) == 1)

# 4
print (multiplicative_persistence(PositiveIntegerInput(4)) == 0)
print (multiplicative_persistence(PositiveIntegerInput(39)) == 3)
print (multiplicative_persistence(PositiveIntegerInput(999)) == 4)
print (multiplicative_persistence(PositiveIntegerInput(10)) == 1)

# 5
print (mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0)
print (mse(VectorPairInput([1, 2], [3, 4])) == 4.0)

# 6
print (prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)")
print (prime_factorization(PositiveIntegerInput(13)) == "(13)")

# 7
print (pyramid(PositiveIntegerInput(1)) == 1)
print (pyramid(PositiveIntegerInput(5)) == 2)
print (pyramid(PositiveIntegerInput(14)) == 3)
print (pyramid(PositiveIntegerInput(2)) == "It is impossible")

# 8
print (is_balanced_number(PositiveIntegerInput(1234006)) is True)
print (is_balanced_number(PositiveIntegerInput(123456)) is False)
print (is_balanced_number(PositiveIntegerInput(7)) is True)