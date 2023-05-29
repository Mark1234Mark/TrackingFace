import cv2

# Загрузка классификатора лица
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Инициализация видеопотока
video_capture = cv2.VideoCapture(0)

# Цвет зеленой рамки (BGR)
green_color = (0, 255, 0)

while True:
    # Считывание кадра из видеопотока
    ret, frame = video_capture.read()

    # Конвертация в оттенки серого для обнаружения лица
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Отрисовка зеленой рамки вокруг каждого обнаруженного лица
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), green_color, 2)

    # Отображение результата
    cv2.imshow('Face Tracking', frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()
