<h3>Введение</h3>
Данный ноутбук, и система тестирования разработана мной для прохождения учениками практических заданий из курса Основы статистики 2 часть Карпова на Stepik.
В оригинальном курсе практика выполняется на R, поэтому пользователи Python не могут пройти практическую часть курса. 
Эта работа исправляет эту проблему.

Cсылка на курс Stepik:
https://stepik.org/524

Оригинальные задания курса предполагают написание кода на R, поэтому после успешного прохождения задания на Python данная тестирующая система предоставит код на R для дальнейшей вставки в тестирующую систему на Stepik. Таким образом, можно синхронизировать прогресс выполнения и получать баллы на курсе Stepik не изучая специально R.

Как проходить задания:
- Читаем задание шага

- Пишем ваше решение в специальную заготовку функции (обратите внимание на форматы ввода/вывода). Запускаем ячейку.

- Запускаем тестирующий блок для проверки вашего решения

После успешного прохождения теста, выводится код на R для вставки в систему Stepik.

<h3>1. Анализ номинативных данных </h3>
<a href=https://github.com/maryginm/Statistic_part2/blob/1f1dcc0859b5ec226d677c7e4993435240171f50/Part%201.ipynb>Практика 1. Ноутбук с заданиями</a>

Ссылка на оригинальные задания:
https://stepik.org/lesson/26186/step/1?unit=8128

<h3>Описание файлов части 1:</h3>

- Part 1.ipynb - Jupyter notebook. Тут читаем задания и пишем свои решения.
- Part1.py - Python code. Тестирующий модуль части. Модуль сравнивает ответ пользовательской функции с ответом проверяющей функции.
- testdata.p1 - объект pickle (словарь с тестовыми данными.)
- answers.p1 - объект pickle (словарь с ответами на R для вставки в проверяяющую систему Stepik.)
- diamonds.csv - данные для подгрузки в ноутбук (для тестовой системы не требуются, нужны для удобства написания кода в ноутбуке)
- insert.png - изображение иллюстрация
