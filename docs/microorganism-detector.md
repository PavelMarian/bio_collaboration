## Модель детекции микроорганизмов

Модель Faster R-CNN ResNet-50 FPN v2 находит на микроскопических снимках активного ила семь классов микроорганизмов. Она подключается блоком `models.microorganism_detection` в `config.coco.example.yaml` и загружается существующим `TorchvisionCocoModel`. Модель обучена на кадрах JPG и BMP размером 1280 x 960.

| label | класс |
| --- | --- |
| 1 | `attached_ciliates` |
| 2 | `filamentous_bacteria` |
| 3 | `rotifers` |
| 4 | `testate_amoebae` |
| 5 | `free_swimming_ciliates` |
| 6 | `nematoda` |
| 7 | `gastrotrichs` |

Label 0 означает фон. Файл категорий `annotations/microorganism_instances.json` входит в репозиторий. Id категорий в нём равны label модели. Загрузчик сопоставляет label k с k-й категорией по возрастанию id.

### Веса

Веса прикреплены к выпуску [microorganism-detector-v0.1.0](https://github.com/PavelMarian/bio_collaboration/releases/tag/microorganism-detector-v0.1.0) и в git не хранятся. Команды выполняются из корня репозитория:

```bash
mkdir -p models
curl -L -o models/microorganisms_fasterrcnn.pt https://github.com/PavelMarian/bio_collaboration/releases/download/microorganism-detector-v0.1.0/microorganisms_fasterrcnn.pt
curl -L -o models/microorganisms_fasterrcnn.pt.sha256 https://github.com/PavelMarian/bio_collaboration/releases/download/microorganism-detector-v0.1.0/microorganisms_fasterrcnn.pt.sha256
(cd models && shasum -a 256 -c microorganisms_fasterrcnn.pt.sha256)
```

Ожидаемый SHA-256: `0355b1ca23ad3cc88483d06de751f81b64a6a648f6888a2ff4d22fbb95e09af9`. На Linux вместо `shasum -a 256 -c` подходит `sha256sum -c`. Файл с другой суммой не используется.

### Настройки

- Блок `models.microorganism_detection`: `backend: torchvision`, `architecture: fasterrcnn_resnet50_fpn_v2`, `weights: models/microorganisms_fasterrcnn.pt`, `coco_annotations: annotations/microorganism_instances.json`. Пути считаются от каталога YAML-файла.
- `inference.microorganism_confidence: 0.20` — порог принятия детекции, выбранный на validation.
- `min_size` 960 и `max_size` 1280 загрузчик берёт из checkpoint. Внутренние параметры детектора остаются значениями torchvision: score от `0.05`, NMS IoU `0.50`, не больше 100 детекций на снимок.
- `inference.image_size` и `inference.iou_threshold` на этот детектор не влияют. Общий `max_detections` ограничивает принятые детекции и маски флокул; значение 300 не меньше собственного предела детектора.

### Запуск

Полный запуск требует весов модели флокул. В примере это `models/floc_maskrcnn.pt` с категориями `annotations/floc_instances.json`. Эти файлы в репозиторий не входят. Без них команда завершается ошибкой `FileNotFoundError: Missing floc segmentation model`.

```bash
python cli.py --validate-models --config config.coco.example.yaml
python cli.py <каталог снимков> --config config.coco.example.yaml --output <каталог результатов>
```

После установки проекта вместо `python cli.py` можно вызывать `active-sludge-cv` с теми же аргументами. Детекции записываются в `microorganism_detections.json`, счётчики — в `microorganism_counts.csv` и `combined_analysis.csv`. Счётчики строятся по тем же детекциям.

### Ограничения

- Исследовательский прототип: модель обучена одним запуском `frcnn_bn_frozen` на небольшом наборе.
- macro F1 по объектам при IoU 0,50 и пороге 0,20: 0,324 на validation (46 снимков, 172 объекта) и 0,304 на test (45 снимков, 162 объекта, однократная оценка). F1 — мера совпадения рамок и классов с разметкой, а не доля верно распознанных снимков.
- На test модель не нашла ни одной из 9 размеченных нематод.
- Нитчатые бактерии модель пересчитывает: на test в среднем на 2,3 объекта на снимок больше разметки.
- Разметка неполна: часть детекций приходится на неразмеченные объекты.
- Разбиение невременное: test содержит кадры тех же архивов, что и train. Проверки на новых пробах, датах съёмки и микроскопах не было.
- Confidence — оценка детектора, а не калиброванная вероятность.
- Полный запуск вместе с настоящей моделью флокул не проверялся.
