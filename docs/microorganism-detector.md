## Модель детекции микроорганизмов

Модель Faster R-CNN ResNet-50 FPN v2 находит на микроскопических снимках активного ила семь классов микроорганизмов. Она подключается блоком `models.microorganism_detection` в `config.coco.example.yaml` и загружается существующим `TorchvisionCocoModel`. Текущие веса получены запуском `frcnn_bn_frozen_mdas2025_finetune` и рассчитаны на кадры JPG и BMP размером 1280 x 960.

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

Веса прикреплены к предварительному выпуску [microorganism-weights-v2](https://github.com/PavelMarian/bio_collaboration/releases/tag/microorganism-weights-v2) архивом 7z и в git не хранятся. В архиве два файла: веса `microorganisms_fasterrcnn_finetune.pt` и их сумма `SHA256SUMS`. Для распаковки нужен консольный 7-Zip `7zz`, на macOS он ставится командой `brew install sevenzip`. Команды выполняются из корня репозитория:

```bash
mkdir -p models
curl -L -o models/microorganisms_fasterrcnn_finetune.7z https://github.com/PavelMarian/bio_collaboration/releases/download/microorganism-weights-v2/microorganisms_fasterrcnn_finetune.7z
curl -L -o models/microorganisms_fasterrcnn_finetune.7z.sha256 https://github.com/PavelMarian/bio_collaboration/releases/download/microorganism-weights-v2/microorganisms_fasterrcnn_finetune.7z.sha256
(cd models && shasum -a 256 -c microorganisms_fasterrcnn_finetune.7z.sha256)
7zz x -omodels models/microorganisms_fasterrcnn_finetune.7z
(cd models && shasum -a 256 -c SHA256SUMS)
```

SHA-256 архива: `d771b1124ccd042da8fb8862ed14d26c80956f787abf863c74390e832258c18b`. SHA-256 весов `models/microorganisms_fasterrcnn_finetune.pt`: `5fd5067b06178e3472e020a676c2aed32d164957e35036b083d7809e1e5b552d`. На Linux вместо `shasum -a 256 -c` подходит `sha256sum -c`. Файл с другой суммой не используется.

### Настройки

- Блок `models.microorganism_detection`: `backend: torchvision`, `architecture: fasterrcnn_resnet50_fpn_v2`, `weights: models/microorganisms_fasterrcnn_finetune.pt`, `coco_annotations: annotations/microorganism_instances.json`. Пути считаются от каталога YAML-файла.
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

- Исследовательский прототип: веса получены одним запуском дообучения на небольшом наборе.
- macro F1 по объектам при IoU 0,50 и пороге 0,20: 0,411966 на validation с исправленной разметкой (46 кадров, 183 объекта). Порог выбран на этой же validation, поэтому оценка оптимистична: при порогах 0,15 и 0,25 macro F1 равен 0,380 и 0,348. F1 — мера совпадения рамок и классов с разметкой, а не доля верно распознанных снимков.
- На test эти веса не оценивались. Значения 0,324 на validation и 0,304 на test относятся к прежним весам из выпуска microorganism-detector-v0.1.0.
- На validation модель пропустила 105 из 183 размеченных объектов. Слабее всего классы `free_swimming_ciliates` (F1 0,087, 5 объектов), `attached_ciliates` (0,234, найдено 16 из 72) и `filamentous_bacteria` (0,303). У `nematoda` всего 3 объекта, их F1 неустойчив.
- Нитчатые бактерии модель пересчитывает: на validation 98 рамок при 54 размеченных объектах.
- Полнота разметки не подтверждена: часть лишних рамок может приходиться на неразмеченные объекты.
- Разбиение невременное: 41 из 46 кадров validation взяты из тех же архивов, что и кадры train. Проверки на новых пробах, датах съёмки и микроскопах не было.
- Confidence — оценка детектора, а не калиброванная вероятность.
- Подключение проверено на установленном проекте без исследовательского кода: `TorchvisionCocoModel` строго загружает веса, порядок классов и порог 0,20 совпадают с проверенным экспортом, на 8 кадрах validation детекции и счётчики совпали с эталоном.
- Полный запуск вместе с моделью флокул проверен только на этих 8 кадрах. Модель `floc_mask_model.pt` из выпуска microorganism-detector-v0.1.0 подключалась блоком `backend: ultralytics`, остальные настройки взяты из `config.coco.example.yaml`. Детекции микроорганизмов совпали с эталоном, результаты флокул ни с чем не сверялись.

### Данные и лицензии

- Кадры проекта с исправленной разметкой.
- Внешний набор MD-AS-2025, Dataset A, кадры категорий Arcellinida, Digononta, Monogononta и Nematoda: Pang Z., Wang A., Chen Y., Xu T., Shao Z., Ge C., He C., Tao Y. Image Dataset of Microfauna in Activated Sludge. Scientific Data 12, 1972 (2025), doi:10.1038/s41597-025-06228-6. Данные: Zenodo, doi:10.5281/zenodo.15210103, лицензия CC BY 4.0.
- Начальные веса: torchvision `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`.
