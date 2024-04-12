# kweRu — сервис для определения ключевых слов и УДК статьи

Проект представляет собой сервис, который позволяет автоматически генерировать ключевые слова и УДК (Универсальная десятичная классификация) по тексту аннотации статьи. Инструмент полезен для авторов научных работ, желающих быстро определить ключевые тематики своих публикаций.

## Функционал

- Автоматическое извлечение ключевых слов из текста аннотации.
- Отбор подходящих УДК на основе содержания аннотации.
- Интуитивно понятный интерфейс для ввода текста аннотации и управления настройками.
- Возможность задания количества ключевых слов (0-15) для извлечения.
- Поддержка сохранения истории запросов при перезагрузке страницы.

## Технологии

- **Frontend**: Реализован с использованием React.js и библиотеки Material-UI для создания пользовательского интерфейса.
- **Backend**: Написан на Python с использованием фреймворка FastAPI для обработки запросов от пользователей.
- **Деплоймент**: Для развертывания проекта используется Docker.

## Установка и запуск

1. **Клонировать репозиторий**:

```bash
git clone https://github.com/Dream-Team-Studcamp/udc.git
cd udc
```

2. **Запустить проект**:

```bash
docker-compose up --build
```

## Разработчики

- Мельгизин Марат - [m-melgizin](https://github.com/m-melgizin)
- Мясников Александр - [HermannStettin](https://github.com/HermannStettin)
- Никифорова Анна - [AnnaNik334743](https://github.com/AnnaNik334743)

