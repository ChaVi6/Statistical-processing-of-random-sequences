import math
import numpy as np
import random
import statistics
from numpy import sqrt, pi, e
from prettytable import PrettyTable
import file
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot
from scipy.stats import skew, kurtosis, arcsine, uniform, norm, chi2_contingency, ks_2samp

# 1.1. Считываем выборку x из файла. Создаем на ее основе 10 подвыборок

n = file.N
x = file.x


def create_10(x):
    x_podv = []
    x_perm = x.copy()
    random.shuffle(x_perm)
    for i in range(10):
        start = int((i * n) / 10)
        end = int(((i + 1) * n) / 10)
        chunk = x_perm[start:end]
        x_podv.append(chunk)
    return x_podv


# 1.2. Представим визуально оценки функции плотности распределения

# 1.2.1. Построим выборочную функцию распределения

def sdf(x):
    x_sorted = sorted(x)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    pyplot.step(x_sorted, y)
    pyplot.xlabel('Значения x')
    pyplot.ylabel('Распределение веротяностей')
    pyplot.show()

# 1.2.2. Построим абсолютную и относительную гистограммы

def plot_histograms(x):
    pyplot.hist(x, bins=10)
    pyplot.xlabel('X')
    pyplot.ylabel('F(x)')
    pyplot.title('Абсолютная гистограмма')
    pyplot.show()

    pyplot.hist(x, bins=10, density=True)
    pyplot.xlabel('X')
    pyplot.ylabel('F(x)')
    pyplot.title('Относительная гистограмма')
    pyplot.show()

# 1.2.3. Построим оценки плотности с применением ядерного оценивания

linespace = np.linspace(8, 22, 50)

def kde(sample, x, kernel='gaussian', bandwidth=0.7) -> np.ndarray:
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde.fit(np.asarray(sample)[:, np.newaxis])
    log_pdf = kde.score_samples(x[:, np.newaxis])
    return np.exp(log_pdf)

def kde_prob(sample, linespace, kernel='gaussian', bandwidth=0.7, **kwargs):
    kwargs['label'] = kwargs.get('label', 'kde')
    density = kde(sample, linespace, kernel, bandwidth)
    pyplot.plot(linespace, density, **kwargs)

fig_1, ax = pyplot.subplots(1, 1)
ax.set_title('Kernel Density Estimation\nBandwidth = 0.7')
kde_prob(x, linespace, kernel="gaussian", bandwidth=0.7, label="Gaussian", color='red')
kde_prob(x, linespace, kernel="tophat", bandwidth=0.7, label="Tophat", color='green')
kde_prob(x, linespace, kernel="linear", bandwidth=0.7, label="Linear", color='black')
kde_prob(x, linespace, kernel="exponential", bandwidth=0.7, label="Exponential", color='blue')
pyplot.hist(x, bins=10, density=True, color='yellow')
fig_1.legend()
pyplot.show()

# 1.3. Определим точечные оценки

def find_moment(x, degree, mean):  # вспомогательная функция для вычисления центрального момента
    res = 0
    for i in range(len(x)):
        res += (x[i] - mean) ** degree
    return round(res / len(x), 3)


def point_estimates(x, true_table, true_plot):
    x_podv = create_10(x)  # выборка
    x_mean = []  # среднее арифметическое
    x_med = []  # медиана
    x_sr = []  # средина размаха
    var2 = []  # дисперсия
    var = []  # отклонение дисперсий
    m3 = []  # третий центральный момент
    m4 = []  # четвёртый центральный момент
    ass = []  # асимметрия
    ex = []  # эксцесс
    num = []

    for i in range(len(x_podv) + 1):
        if i == 10:
            a = x
            num.append(n)
        else:
            a = x_podv[i]
            num.append(int(n / 10))
        x_mean.append(round(statistics.mean(a), 3))
        x_med.append(round(statistics.median(a), 3))
        x_sr.append(round((min(a) + max(a)) / 2, 3))
        var2.append(round(statistics.variance(a), 3))
        var.append(round(var2[i] ** (1 / 2), 3))
        m3.append(find_moment(a, 3, x_mean[i]))
        m4.append(find_moment(a, 4, x_mean[i]))
        ass.append(skew(a))
        ex.append(kurtosis(a))

    if true_table:
        table = PrettyTable(['Размер подвыборки', 'x_mean', 'x_med', 'x_sr', 'var2', 'var', 'm3', 'm4', 'As', 'Ex'])
        for j in range(len(x_podv) + 1):
            table.add_row([num[j], x_mean[j], x_med[j], x_sr[j], var2[j], var[j], m3[j], m4[j], ass[j], ex[j]])
        table.align = "c"

        print(table)
        print("Мода =", round(statistics.mode(x), 3))

    lower_bound = round(statistics.mean(x) - statistics.stdev(x) * 1.9602111525053565, 3)
    upper_bound = round(statistics.mean(x) + statistics.stdev(x) * 1.9602111525053565, 3)

    print('Интерквартильный промежуток: [', lower_bound, ', ', upper_bound, ']')

    y = np.zeros(10)
    y1 = np.ones(10)
    y2 = y1 + 1

    if true_plot:
        pyplot.title('Среднее арифметическое, медиана, средина размаха')
        pyplot.scatter(x_mean[:10], y1, label='ср. арифм.')
        pyplot.scatter(x_mean[10], y1[0], label='ср. арифм. (n = 6000)', marker='d')
        pyplot.scatter(x_med[:10], y, label='медиана')
        pyplot.scatter(x_med[10], y[0], label='медиана (n = 6000)', marker='d')
        pyplot.scatter(x_sr[:10], y2, label='средина размаха')
        pyplot.scatter(x_sr[10], y2[0], label='средина размаха (n = 6000)', marker='d')
        pyplot.legend(loc='upper right')
        pyplot.show()

        pyplot.title('Дисперсия')
        pyplot.scatter(var2[:10], y, label='дисперсия')
        pyplot.scatter(var2[10], y[0], label='дисперсия (n = 6000)', marker='d')
        pyplot.legend()
        pyplot.show()

        pyplot.title('Третий центральный момент')
        pyplot.scatter(m3[:10], y, label='третий центральный момент')
        pyplot.scatter(m3[10], y[0], label='третий центральный момент (n = 6000)', marker='d')
        pyplot.legend()
        pyplot.show()

        pyplot.title('Четвертый центральный момент')
        pyplot.scatter(m4[:10], y, label='четвертый центральный момент')
        pyplot.scatter(m4[10], y[0], label='четвертый центральный момент (n = 6000)', marker='d')
        pyplot.legend()
        pyplot.show()

        pyplot.title('Асимметрия')
        pyplot.scatter(ass[:10], y, label='асимметрия')
        pyplot.scatter(ass[10], y[0], label='асимметрия (n = 6000)', marker='d')
        pyplot.legend()
        pyplot.show()

        pyplot.title('Эксцесс')
        pyplot.scatter(ex[:10], y, label='эксцесс')
        pyplot.scatter(ex[10], y[0], label='эксцесс (n = 6000)', marker='d')
        pyplot.legend()
        pyplot.show()

    return [x_mean[0], x_mean[10], var2[0], var2[10]]

# 1.4. Определяем интервальные оценки

def calculate_standard_deviation(x, x_mean):
    n = len(x)
    deviations = [(xi - x_mean) ** 2 for xi in x]
    variance = sum(deviations) / (n - 1)
    standard_deviation = math.sqrt(variance)
    return standard_deviation


def interval_estimates(x):
    # Первый момент
    def first_moment(x):
        x_mean = statistics.mean(x)
        s = calculate_standard_deviation(x, x_mean)  # стандартное отклонение для x и x_mean
        n = len(x)
        if n == 6000:  # для выборки
            k = 1.2816927023905091
        if n == 600:  # для подвыборок
            k = 1.282966488016581
        return [x_mean - k * (1 / n ** (1 / 2)) * s, x_mean + k * (1 / n ** 1 / 2) * s]

    # Второй момент
    def second_moment(x):
        x_mean = statistics.mean(x)
        s = calculate_standard_deviation(x, x_mean)  # стандартное отклонение для x и x_mean
        n = len(x)
        if n == 6000:  # для выборки
            k1 = 6139.79652262585
            k2 = 5859.05997068446
        if n == 600:  # для подвыборок
            k1 = 643.763416639911
            k2 = 555.093023265182
        return [s ** 2 * (n - 1) / k1, s ** 2 * (n - 1) / k2]

    # Интерквантильный промежуток

    # параметрический промежуток

    def get_param(x):
        x_sr = statistics.mean(x)
        s = calculate_standard_deviation(x, x_sr)
        lower_bound = float(x_sr - s * 1.9602111)
        upper_bound = float(x_sr + s * 1.9602111525053565)
        return [round(lower_bound, 3), round(upper_bound, 3)]

    # Непараметрические толерантные пределы

    x_podv = create_10(x)
    x_podv.append(x)
    points = point_estimates(x, False, False)
    m1 = []
    m2 = []
    param = []
    nonparam = []
    for i in range(len(x_podv)):  # 0-10
        if i != len(x_podv) - 1:  # != 10
            nonparam.append("")
        else:
            nonparam.append([min(x_podv[i]), max(x_podv[i])])
        param.append(get_param(x_podv[i]))
        m1.append(first_moment(x_podv[i]))
        m2.append(second_moment(x_podv[i]))
    pyplot.title("1й момент, полная выборка")
    pyplot.scatter(points[1], [0], color="red", label="точечная оценка")
    pyplot.plot(m1[10], [0, 0], label="интервальная оценка")
    pyplot.legend()
    pyplot.show()

    pyplot.title("1й момент, подвыборка")
    pyplot.scatter(points[0], [0], color="red", label="точечная оценка")
    pyplot.plot(m1[0], [0, 0], label="интервальная оценка")
    pyplot.legend()
    pyplot.show()

    print('Интервальные оценки для 1ого момента:', m1[10])
    for i in range(10):
        print('Подвыборка №', i+1, m1[i] )

    pyplot.title("дисперсия, полная выборка")
    pyplot.scatter(points[3], [0], color="red", label="точечная оценка")
    pyplot.plot(m2[10], [0, 0], label="интервальная оценка")
    pyplot.legend()
    pyplot.show()

    pyplot.title("дисперсия, подвыборка")
    pyplot.scatter(points[2], [0], color="red", label="точечная оценка")
    pyplot.plot(m2[0], [0, 0], label="интервальная оценка")
    pyplot.legend()
    pyplot.show()

    print('\nИнтервальные оценки для дисперсии:', m2[10])
    for i in range(10):
        print('Подвыборка №', i+1, m2[i] )

    pyplot.title("полная выборка, непараметрические пределы")
    pyplot.plot(nonparam[10], [0, 0])
    pyplot.show()

    pyplot.title("подвыборка, параметрические пределы")
    pyplot.plot(param[0], [0, 0])
    pyplot.show()

    print('\nПараметрические пределы для подвыборок:')
    for i in range(10):
        print('Подвыборка №', i+1, param[i] )

#interval_estimates(x)

# 2.  Идентификация закона и параметров распределения

# 2.1.Выполним сравнение функций плотностей с гистограммой

def compare(x):
    x_theory = np.linspace(8, 22, 50)

    mu, sigma = norm.fit(x) # параметры равномерного распределения
    normal = norm.pdf(x_theory, mu, sigma)
    pyplot.plot(x_theory, normal)
    pyplot.xlabel('Значение')
    pyplot.ylabel('Плотность вероятности')
    pyplot.title('Нормальное распределение')
    pyplot.hist(x, bins=10, density=True)
    pyplot.show()

    a, b = uniform.fit(x) # параметры равномерного распределения
    uni = uniform.pdf(x_theory, a, b)
    pyplot.xlabel('Значение')
    pyplot.ylabel('Плотность вероятности')
    pyplot.title('Равномерное распределение')
    pyplot.plot(x_theory, uni)
    pyplot.hist(x, bins=10, density=True)
    pyplot.show()

    x_theory = np.linspace(7.9995, 22.0005, 50)
    c, scale = arcsine.fit(x) # параметры дугового синуса
    arcsin = arcsine.pdf(x_theory, c, scale)
    pyplot.xlabel('Значение')
    pyplot.ylabel('Плотность вероятности')
    pyplot.title('Распределение арксинус')
    pyplot.plot(x_theory, arcsin)
    pyplot.hist(x, bins=10, density=True)
    pyplot.show()

# 2.2. Определим параметры теоретических распределений

def compare_with_theory_params(x):

    sr = statistics.mean(x) # среднее значение
    var = statistics.variance(x) # дисперсия
    x_sorted = sorted(x)
    y = np.linspace(0, 1, 6000)

    def moment_method_norm(): # метод моментов нормального распределения
        a = sr
        sigma = var ** (1/2)
        s = "a = " + str(a) + ", sigma = " + str(sigma)

        pyplot.title('Нормальное распределение')
        pyplot.step(x_sorted, y, label='Эмпирическая функция')

        num = np.linspace(8, 22, 6000)
        theor = sorted(norm.cdf(x, a, sigma))
        pyplot.plot(num, theor, label='Теоретическая функция')
        pyplot.legend()
        pyplot.show()
        return s

    def moment_method_uni(): # метод моментов равномерного распределения
        a = sr - (3 * sr) ** (1 / 2)
        b = (12 * var) ** (1 / 2) + a
        s = "a = " + str(a) + ", scale = " + str(b - a)

        pyplot.title('Равномерное распределение')
        pyplot.step(x_sorted, y, label='Эмпирическая функция')

        num = np.linspace(8, 22, 6000)
        theor = sorted(uniform.cdf(x, a, b))
        pyplot.plot(num, theor, label='Теоретическая функция')
        pyplot.legend()
        pyplot.show()
        return s

    def moment_method_arcsin():
        c = sr
        a = (2 * var) ** (1 / 2)
        s = "a = " + str(a) + ", c = " + str(c)

        pyplot.title('Распределение арксинуса')
        pyplot.step(x_sorted, y, label='Эмпирическая функция')

        num = np.linspace(7.9995, 22.0005, 50)
        theor = sorted(arcsine.cdf(num, a, c))
        pyplot.plot(num, theor, label='Теоретическая функция')
        pyplot.legend()
        pyplot.show()
        return s

    def mmp_norm():
        a, sigma = norm.fit(x)
        s = 'a = ' + str(a) + ', sigma = ' + str(sigma)
        num = np.linspace(8, 22, 6000)
        theor = norm.pdf(num, a, sigma)
        pyplot.title('Нормальное распределение')
        pyplot.plot(num, theor)
        pyplot.hist(x, bins=10, density=True)
        pyplot.show()
        return s

    def mmp_uni():
        a, b = uniform.fit(x)
        s = 'a = ' + str(a) + ', scale = ' + str(b)
        num = np.linspace(8, 22, 6000)
        theor = uniform.pdf(num, a, b)
        pyplot.title('Равномерное распределение')
        pyplot.plot(num, theor)
        pyplot.hist(x, bins=10, density=True)
        pyplot.show()
        return s

    def mmp_arcsin():
        c, a = arcsine.fit(x)
        s = 'a = ' + str(c) + ', c = ' + str(a)
        num = np.linspace(7.9995, 22.0005, 50)
        theor = arcsine.pdf(num, c, a)
        pyplot.title('Распределение арксинус')
        pyplot.plot(num, theor)
        pyplot.hist(x, bins=10, density=True)
        pyplot.show()
        return s

    print('Нормальное распределение:')
    print('метод моментов:', moment_method_norm())
    print('ммп:', mmp_norm())

    print('\nРавномерное распределение:')
    print('метод моментов:', moment_method_uni())
    print('ммп:', mmp_uni())

    print('\nРаспределение арксинуса:')
    print('метод моментов:', moment_method_arcsin())
    print('ммп:', mmp_arcsin())

# 2.3. Произведем проверку гипотез

def hi_square(x, theor):
    observed = []
    for i in range(len(x)):
        if (x[i] * theor[i] == 0):
            observed.append([x[i], 0.000000000000001])
        else:
            observed.append([x[i], theor[i]])
    chi2, p, dof, expected = chi2_contingency(observed)
    print(chi2, p)


def kolmogorov_smirnov(x, theor):
    ks_statistic, p_value = ks_2samp(x, theor)
    print(ks_statistic, p_value)


def von_mises(x, a, b):
    mises = 1 / (12 * len(x))
    f = arcsine.cdf(x, a, b)
    for i in range(0, len(x)):
        mises += (f[i] - (2 * i - 1) / (2 * n)) ** 2
    print("mises:", mises / n)