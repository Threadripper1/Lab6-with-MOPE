from _pydecimal import Decimal, ROUND_UP, ROUND_FLOOR
from IPython.display import HTML, display
import random
import numpy as np
# import numpy as np
from scipy.stats import f, t, ttest_ind, norm
class Criteries:
    @staticmethod
    def get_cohren_value(size_of_selections,qty_of_selections, significance ):
        size_of_selections += 1
        partResult1 = significance/(size_of_selections-1)
        params = [partResult1, qty_of_selections, (size_of_selections-1-1)*qty_of_selections]
        fisher = f.isf(*params)
        result = fisher/(fisher+(size_of_selections-1-1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    @staticmethod
    def get_student_value(f3,significance):
        return Decimal(abs(t.ppf(significance/2,f3))).quantize(Decimal('.0001')).__float__()

    @staticmethod
    def get_fisher_value(f3,f4,significance):
        return Decimal(abs(f.isf(significance,f4,f3))).quantize(Decimal('.0001')).__float__()
cr = Criteries()

def dob(*args):
    res = [1 for _ in range(len(args[0]))]
    for i in range(len(args[0])):
        for j in args:
            res[i] *= j[i]
    return res

def getcolumn(arr, n):
    return [i[n] for i in arr]

def getb(xs, ys):
    forb = [getcolumn(xs, i) for i in range(len(xs[0]))]
    ms = [[sum(dob(i, j))/N for j in forb] for i in forb]
    k = [sum(dob(ys, i))/N for i in forb]
    numerators = [[ms[i][0:j] + [k[i]] + ms[i][j+1:] for i in range(len(ms))] for j in range(len(ms))]
    return [np.linalg.det(i)/np.linalg.det(ms) for i in numerators]

def getXnat(Xextr, Xnorm, level=0):
    res = [[i[j]*(Xextr[j][1] - sum(Xextr[j])/2)+sum(Xextr[j])/2 for j in range(len(i))] for i in Xnorm]
    if level == 0:
        return res
    for i in range(len(res)):
        res[i]+= [res[i][1]*res[i][2],
                    res[i][1]*res[i][3],
                    res[i][2]*res[i][3],
                    res[i][1]*res[i][2]*res[i][3]]
        if level == 2:
            res[i] += [res[i][1]**2,
                         res[i][2]**2,
                         res[i][3]**2]
    return res
def getY(x):
    bb = [0.4, 0.3, 7.0, 6.9, 1.5, 0.5, 0.8, 2.0, 0.4, 8.1, 8.7]
    return sum(dob(x, bb)) + random.randrange(0, 1000
                                              ) - 5

Xextr = ((1, 1), (-40, 20), (-70, -10), (-10, -20))
Yextr = (200 + sum([i[0] for i in Xextr]), 200 + sum([i[1] for i in Xextr]))
m = 3
N = 15
p = 0.95
x0123n = (
    (1, -1, -1, -1),
    (1, -1, -1, 1),
    (1, -1, 1, -1),
    (1, -1, 1, 1),
    (1, 1, -1, -1),
    (1, 1, -1, 1),
    (1, 1, 1, -1),
    (1, 1, 1, 1),
    (1, -1.215, 0, 0),
    (1, 1.215, 0, 0),
    (1, 0, -1.215, 0),
    (1, 0, 1.215, 0),
    (1, 0, 0, -1.215),
    (1, 0, 0, 1.215),
    (1, 0, 0, 0)
)
Xnat = getXnat(Xextr, x0123n, 2)
Y = list(list(getY(Xnat[i]) for j in range(m)) for i in range(N))
while True:
    Yavg = [sum(i) / len(i) for i in Y]
    S = [sum([(Yavg[j] - Y[j][i]) ** 2 for i in range(m)]) / m for j in range(N)]
    Gp = max(S) / sum(S)
    F1 = m - 1
    F2 = N
    q = 1 - p
    Gt = cr.get_cohren_value(F2, F1, q)
    print("Y:", Y)
    print("Yavg:", Yavg)
    print("Критерій Кохрена\nGp = %s    Gt = %s" % (Gp, Gt))
    if Gp < Gt:
        print("Дисперсія однорідна.")
        break
    else:
        print("Отримали неоднорідну дисперсію, збільшуємо m.")
        for i in range(len(Y)):
            m += 1
            Y[i].append(getY(Xnat[i]))
level = 2
S2_B = sum(Yavg) / N
while True:
    Xcode = getXnat(((1, 1), (-1, 1), (-1, 1), (-1, 1)), x0123n, level)
    Xnat = getXnat(Xextr, x0123n, level)
    b = getb(Xnat, Yavg)
    yr = [sum(dob(b, i)) for i in Xnat]
    print('\n\nb:', b)
    print("Значення Y для рівняння з отриманими коефіцієнтами:", yr)
    S2_beta = S2_B / (N * m)
    beta = [sum(dob(getcolumn(Xcode, i), Yavg)) / N for i in range(len(Xnat[0]))]
    ts = [abs(i) / ((S2_beta) ** (1 / 2)) for i in beta]
    F3 = F1 * F2
    tt = cr.get_student_value(F3, q)
    bzn = [0 if ts[i] < tt else b[i] for i in range(len(b))]

    d = len(ts) - ts.count(0)
    yzn = [sum(dob(bzn, i)) for i in Xnat]
    print("\n\nКритерій Стьюдента:")
    print("tt = %s   t = %s" % (tt, ts))
    print("d = %s    Значимі коефіцієнти: %s" % (d, bzn))
    print("Значення Y враховуючи лише значимі коефіцієнти:", yzn)

    # Fisher
    print("\n\nКритерій Фішера:")
    Sad = m * sum([(yzn[i] - Yavg[i]) ** 2 for i in range(len(Yavg))]) / (N - d)
    F4 = N - d
    Ft = cr.get_fisher_value(F3, F4, q)
    Fp = Sad / S2_B
    print("Sad = %s    Ft = %s    Fp = %s" % (Sad, Ft, Fp))
    if Fp < Ft or level == 2:
        break
    else:
        print("Отримане рівняння регресії - неадекватне. Збільшуємо к-сть членів ряду")
        level += 1
print("Отримане рівняння регресії - адекватне.") if Fp < Ft else print("Отримане рівняння регресії - неадекватне.")
print("\n\nСередні/отримані/отримані лише зі значимими коефіцієнтами значення Y")
display(HTML(
    '<table><tr>{}</tr></table>'.format(
        '</tr><tr>'.join(
            '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in [Yavg, yr, yzn])
    )
))

data = [[i + 1] + [round(j, 4) for j in Xnat[i]] + [j for j in Y[i]] + [round(Yavg[i], 4)] for i in range(15)]
foroutpX = 'x<sub>0</sub> x<sub>1</sub> x<sub>2</sub> x<sub>3</sub> '
foroutpX += 'x<sub>1</sub>x<sub>2</sub> x<sub>1</sub>x<sub>3</sub> x<sub>2</sub>x<sub>3</sub> x<sub>1</sub>x<sub>2</sub>x<sub>3</sub> ' if level >= 1 else ''
foroutpX += 'x<sub>1</sub><sup>2</sup> x<sub>2</sub><sup>2</sup> x<sub>3</sub><sup>2</sup> ' if level >= 2 else ''
for i in range(m):
    foroutpX += "y<sub>%s</sub> " % (i + 1)
foroutpX += 'Y<sub>avg<sub>'
data.insert(0, ["№"] + list(foroutpX.split()))
display(HTML(
    '<table><tr>{}</tr></table>'.format(
        '</tr><tr>'.join(
            '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
    )
))