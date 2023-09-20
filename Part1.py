#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import seaborn as sns
import io
import pickle
from scipy import stats
from PIL import Image
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt


class Test:
    '''Тестирующий класс:
    new_test = Test('test1')
    new_test(function_to_test)

    -> [results of testing]'''

    IMGS = ('1.08',) # Перечень заданий с проверкой картинки
    
    def __init__(self, question: str):
        # Номер вопроса
        self.__question = question

        # Флаг проверки соответствия ответов
        self.__check = False

        # Словарь с тетовыми данными
        self.__tests = {}
        
    def __call__(self, foo: object):
        # Проверка типа задания
        if self.__question in self.IMGS:
            # Если задание графическое тогда тестируем через граф тесты
            self.__test_img(foo)
        else:
            # Иначе тесты на расчет
            self.__test_foo(foo)      
        return self.__get_answer() if self.__check else f'Тесты задания {self.__question} не пройдены'
    
    def __str__(self):
        return f'Тесты задания {self.__question}'
    
    def __get_answer(self):
        """Считывает и возвращает из объекта pickle ответ на задание"""
        with open('answers.p1', 'rb') as f:
            answers = pickle.load(f)
        return answers[self.__question]
    
    def __test_img(self, foo):
        """Тестирует графические задания"""
        self.__load_tests()
        check_list = []
        test_res = Check(self.__question)()(pd.DataFrame(self.__tests[self.__question]))
        foo_test = foo(pd.DataFrame(self.__tests[self.__question]))
        plot_object_1 = io.BytesIO()
        plot_object_2 = io.BytesIO()
        test_res.savefig(plot_object_1)
        foo_test.savefig(plot_object_2)
        plot_object_1.seek(0)
        plot_object_2.seek(0)
        img1 = Image.open(plot_object_1)  # Открываем первое изображение
        control = img1.load()
        img2 = Image.open(plot_object_2)  # Открываем первое изображение
        test = img2.load()
        if img1.size == img2.size:
            x1, y1 = img1.size
            i = 0 #счетчик ризличных пикселей
            # Проходимся последовательно по каждому пикселю картинок
            for x in range(0, x1):
                for y in range(0, y1):
                    if test[x, y] != control[x, y]:
                        # Если пиксель первой картинки по координатах [x,y] не совпадает
                        # с пикселем второй картинки по координатах [x,y], тогда:
                        i = i + 1  # Увеличиваем счетчик на 1
            if i >= x1 * y1 * 0.02:
                check_list.append(False)
                print(f'Картинки разные, количество разных пикселей {i}')
            else:
                check_list.append(True)
                print(f'Тест пройден')
        else:
            print(f'Не соответствуют размеры изображений {img1.size} и {img2.size}')
            check_list.append(False)
        self.__check = all(check_list)
        if self.__check:
            print(f'Все тесты задания {self.__question} пройдены, код на R для вставки:\n\n')
    
    def __test_foo(self, foo):
        '''Тестирует задания на расчет величин'''

        # Считывание тестовых данных
        self.__load_tests()
        check_list = []
        for number, test_data in enumerate(self.__tests[self.__question]):
            test_res = Check(self.__question)()(pd.DataFrame(self.__tests[self.__question][test_data]))
            if test_res != foo(pd.DataFrame(self.__tests[self.__question][test_data])):
                check_list.append(False)
                print(f'Тест {number + 1} не пройден')
                print(self.__tests[self.__question][test_data])
                break
            else:
                check_list.append(True)
                print(f'Тест {number + 1} пройден!')

        # Проверка на корректность всех ответов
        self.__check = all(check_list)
        if self.__check:
            print(f'Все тесты задания {self.__question} пройдены, код на R для вставки:\n\n')
    
    def __load_tests(self):
        '''Загружает тестовые данные из объект pickle в словарь'''
        with open('testdata.p1', 'rb') as f:
            self.__tests = pickle.load(f)

# Тут проверяющие функции
class Check:
    '''
    Класс с проверяющими функциями
    '''
    def __init__(self, question):
        self.question = question
        self.__functions = {'1.01': self.checker191,
                            '1.02': self.checker192,
                            '1.03': self.checker193,
                            '1.04': self.checker194,
                            '1.05': self.checker195,
                            '1.06': self.checker196,
                            '1.07': self.checker197,
                            '1.08': self.checker198}

    def __call__(self):
        return self.__functions[self.question]

    @staticmethod
    def checker191(x: pd.DataFrame):
        a, b = x[0], x[1]
        for x1, x2 in zip(a, b):
            if x1 in ('NA', None) and x2 not in ('NA', None):
                return False
            elif x2 in ('NA', None) and x1 not in ('NA', None):
                return False
        return True

    @staticmethod
    def checker192(x:pd.DataFrame):
        a, b = x[0], x[1]
        count_a = len(set(a)) #градации переменной а
        count_b = len(set(b)) #градации переменной b
        degree = (count_a - 1) * (count_b - 1) #степеней свобод
        table = {} #словарь подсчета
        for el in zip(a, b):
            table[el] = table.get(el, 0) + 1
        if [x for x in table.values() if x < 5]:
            fisher = True
        else:
            fisher = False
        table = pd.DataFrame({'a': a, 'b': b})
        table['val'] = 1
        table = table.pivot_table(index='a', columns='b', values = 'val', aggfunc='sum')
        if fisher:
            if degree == 1:
                pvalue = stats.fisher_exact(table.values)[1]
            else:
                stat, pvalue, deg = stats.chi2_contingency(table.values)[:3]
            return pvalue
        else:
            stat, pvalue, deg = stats.chi2_contingency(table.values)[:3]
            return (stat, deg, pvalue)

    @staticmethod
    def checker193(data:pd.DataFrame):
        res = pd.DataFrame(index=['pvalues'])
        for var in data:
            res[var]=stats.chisquare(data[var].value_counts()).pvalue
        return list(res[res == res.values.min()].dropna(axis=1).columns)

    @staticmethod
    def checker194(data:pd.DataFrame):
        '''Тут ваш код Python'''
        data['important_cases'] = 0
        for var in data:
            if var != 'important_cases':
                data['important_cases'] += data[var] > data[var].mean()
        data['important_cases'] = np.where(data['important_cases'] > 2, 'Yes', 'No')
        return list(data['important_cases'].values)

    @staticmethod
    def checker195(data:pd.DataFrame):
        data['important_cases'] = 0
        numeric = data.select_dtypes('int')
        for el in numeric:
            data['important_cases'] += data[el] > data[el].mean()
        data['important_cases'] = np.where(data['important_cases'] > math.ceil(len(numeric) / 2), 'Yes', 'No')
        return list(data['important_cases'].values)

    @staticmethod
    def checker196(data:pd.DataFrame):
        '''ваш код Python'''
        counts = data[0].value_counts()
        return list(counts[counts == counts.max()].index)

    @staticmethod
    def checker197(data_in:pd.DataFrame):
        counts = pd.DataFrame(data_in.value_counts()).reset_index()
        observed = pd.pivot_table(data=counts, index=counts.columns[1], columns = counts.columns[0], values=0)
        expected = pd.DataFrame(stats.contingency.expected_freq(observed), index=observed.index, columns=observed.columns)
        N = observed.values.sum()
        i = 1 - observed.sum()/N
        j = 1 - observed.sum(axis=1)/N
        coeff = pd.DataFrame(index=j.index, columns=i.index)
        for row in j.index:
            for col in i.index:
                coeff.loc[row, col] = j[row] * i[col]
        ceil = (expected * coeff) ** 0.5
        pierson_stand = (observed - expected) / ceil
        max_element = pierson_stand[pierson_stand == pierson_stand.max().max()].dropna(axis=1, how="all").dropna()
        return [max_element.index[0], max_element.columns[0]]

    @staticmethod
    def checker198(data_in:pd.DataFrame):
        result = plt.figure(figsize=(10, 8))
        sns.histplot(data = data_in, x='color', hue='cut', multiple='dodge', shrink=0.8)
        plt.title('Гистограмма частот')
        return result
