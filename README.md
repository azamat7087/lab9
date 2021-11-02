# lab9
## Распознавание лиц на фото
### Подготовили: Боранбаев Азамат, Уразгалиева Зарина, Кабиев Айбол, Касым Айдос

Данная инструкция предназначена для Linux систем:

Для того, чтобы код запустился необходимо активировать виртуальное окружение в директории с файлом и установить все модули из файла requirements.txt:

`cd lab9`

`python -m venv env`

`source venv/bin/activate`

`pip install requirements.txt`

После установки всех зависимостей нужно загрузить в директорию с python скриптом файлы: 


```
lab9
│   README.md
│   test_dataset    
|   training_dataset
|   lab9.py
|   requirements.txt
|   dlib_face_recognition_resnet_model_v1.dat(Необходимо загрузить отдельно)
|   shape_predictor_68_face_landmarks.dat(Необходимо загрузить отдельно)
```

Скачать эти файлы можно из данного репозитория:
https://github.com/ageitgey/face_recognition_models/tree/master/face_recognition_models/models

Затем нужно запустить скрипт lab9 для начала обучения
