# Telegram-бот для распознавания продуктов питания

 Этот проект представляет собой Telegram-бота, использующего Mask-RCNN для распознавания продуктов на изображениях, подсчета их калорийности и предоставления рецептов на основе распознанных ингредиентов.
 1. Отправьте фотографию продукта боту через Telegram.
 2. Бот вернет распознанные продукты.
 3. Дополнительно можно отредактировать список продуктов, рассчитать по ним калорийность и запросить рецепты приготовления блюд по этим ингредиентам.

### Модель и датасеты

В проекте использовалась библиотека Mask RCNN из репозитория: https://github.com/matterport/Mask_RCNN
Используемый датасет для тестирования взят из Food Recognition Challenge AIcrowd https://www.aicrowd.com/challenges/food-recognition-challenge.
Обученная модель и ее реализация: https://gitlab.aicrowd.com/nikhil_rayaprolu/food-recognition
147k кулинарных рецептов: https://www.povarenok.ru/wiki/pravilorecept и https://www.kaggle.com/datasets/rogozinushka/povarenok-recipes/data
Калорийность продуктов: https://www.kaggle.com/datasets/vaishnavivenkatesan/food-and-their-calories

### Установка и запуск проекта
1. Клонирование репозитория
   
   git clone https://github.com/your-username/food-recognition.git
cd food-recognition

3. Установка зависимостей
Рекомендуемая версия Python 3.7.6

pip install -r requirements.txt

cd Mask_RCNN
python setup.py install

4. Добавить токен вашего телеграм бота в telegram_bot.py (10 строка)
![image](https://github.com/user-attachments/assets/6f448e4e-3b61-4a2b-8dc1-9d624134598c)
5. Запустить телеграм бот
   python telegram_bot.py

   Проект в Kaggle: https://www.kaggle.com/code/iisubbotina/foodscan-telegram-bot-for-food-recognition/notebook
   
