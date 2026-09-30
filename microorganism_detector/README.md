# Детектор микроорганизмов активного ила

Исследовательский прототип. Пакет `sludge_micro` находит на микроскопических снимках активного ила семь групп индикаторных микроорганизмов. Для каждого объекта он возвращает класс, рамку `bbox_xyxy` и confidence, для каждого снимка — число объектов каждого класса. Результат совместим с пайплайном Павла из корня этого репозитория (пакет `cv_module`).

Поставляемая модель — Faster R-CNN ResNet-50 FPN v2 `frcnn_bn_frozen`. Её параметры, SHA-256 весов и результаты оценки записаны в паспорте [model/passport.json](model/passport.json); сводка паспорта приведена в разделе [Паспорт модели](#паспорт-модели). Качество ограничено, см. [Ограничения](#ограничения).

## Состав каталога

- `src/sludge_micro/` — пакет: инференс, проверка модели по паспорту, адаптер и экспорт для пайплайна Павла, оценка, подготовка данных и обучение.
- `model/passport.json` — паспорт поставляемой модели.
- `model/pavel/` — блок конфигурации и файл категорий для пайплайна Павла, совпадающие с экспортом поставляемой модели.
- `configs/microorganisms.yaml` — рабочая конфигурация: классы, правила аудита, подготовки, разбиения и оценки.
- `configs/frcnn_bn_frozen.yaml` — настройки, с которыми обучена поставляемая модель.
- `tests/` — тесты на синтетических изображениях.
- `requirements.lock` — точные версии зависимостей.

Веса модели в Git не хранятся, они прикреплены к выпуску.

## Установка

Проверено на macOS 26.1 (arm64) с CPython 3.13.7 и версиями из `requirements.lock`, вычисления на CPU. Команды выполняются из каталога `microorganism_detector`:

```bash
python3.13 -m venv .venv
.venv/bin/pip install --no-deps -r requirements.lock
.venv/bin/pip install --no-deps --no-build-isolation ..
.venv/bin/pip install --no-deps --no-build-isolation .
```

Вторая команда устанавливает пайплайн Павла из корня репозитория, третья — пакет детектора. Установка скачивает пакеты с PyPI; после неё и загрузки весов инференс работает без сети, исходные веса COCO для него не нужны.

## Получение весов

Веса и их контрольная сумма прикреплены к предварительному выпуску [microorganism-detector-v0.1.0](https://github.com/PavelMarian/bio_collaboration/releases/tag/microorganism-detector-v0.1.0). Файлы кладутся в каталог `weights/`:

```bash
mkdir -p weights
curl -L -o weights/microorganisms_fasterrcnn.pt https://github.com/PavelMarian/bio_collaboration/releases/download/microorganism-detector-v0.1.0/microorganisms_fasterrcnn.pt
curl -L -o weights/microorganisms_fasterrcnn.pt.sha256 https://github.com/PavelMarian/bio_collaboration/releases/download/microorganism-detector-v0.1.0/microorganisms_fasterrcnn.pt.sha256
(cd weights && shasum -a 256 -c microorganisms_fasterrcnn.pt.sha256)
.venv/bin/sludge-micro check-model --passport model/passport.json --weights weights/microorganisms_fasterrcnn.pt
```

`check-model` сверяет размер и SHA-256 файла с паспортом, а также классы, архитектуру, размер выходной головы и параметры масштабирования внутри checkpoint. Если репозиторий станет закрытым, веса скачиваются из браузера под своей учётной записью GitHub или командой `gh release download microorganism-detector-v0.1.0 --repo PavelMarian/bio_collaboration --dir weights` после `gh auth login`; токены в команды и файлы не записываются.

## Запуск

```bash
.venv/bin/sludge-micro predict --passport model/passport.json --weights weights/microorganisms_fasterrcnn.pt --input <каталог снимков> --output <каталог результатов>
```

На вход подаются файлы JPG или BMP из каталога или один файл; модель обучена на кадрах 1280 x 960. Перед предсказанием команда проверяет веса по паспорту и применяет порог confidence из паспорта. Команда запрещает сетевые обращения на время работы.

## Формат результата

- `microorganism_detections.json` — список снимков: `image_id` (относительный путь без расширения), `file_name` и `objects`. Каждый объект содержит `class`, `bbox_xyxy` в пикселях исходного снимка (`x_min`, `y_min`, `x_max`, `y_max`) и `confidence`.
- `microorganism_counts.csv` — `image_id` и по колонке на каждый из семи классов; счётчики построены по тем же объектам, отсутствующие классы имеют ноль.
- `microorganism_detections.parquet` — та же таблица объектов для внутренней обработки.

Confidence — оценка детектора, а не калиброванная вероятность.

## Классы

Порядок label и названия классов приведены в первой таблице [паспорта](#паспорт-модели); label 0 означает фон. Классы вне этих семи (грибы, жгутиковые, эолосомы, голые амёбы, сукторий, детрит) моделью не выдаются.

## Подключение к пайплайну Павла

Экспорт весов, категорий и блока конфигурации:

```bash
.venv/bin/sludge-micro export-pavel --passport model/passport.json --weights weights/microorganisms_fasterrcnn.pt --output <каталог экспорта>
```

Команда записывает `models/microorganisms_fasterrcnn.pt`, `annotations/microorganism_instances.json` и `microorganism_block.yaml`; два последних файла совпадают с `model/pavel/`. Разделы `models.microorganism_detection`, `inference` и `microorganism_classes` блока переносятся в конфигурацию пайплайна, пути в блоке заданы относительно каталога экспорта. `TorchvisionCocoModel` сопоставляет label k с k-й категорией по возрастанию id, а id категорий в файле равны label модели. Параметр `iou_threshold` пайплайна на backend `torchvision` не влияет. Модель флокул подключается в той же конфигурации как обычно; запуск пайплайна:

```bash
python cli.py <каталог снимков> --config <конфигурация пайплайна> --output <каталог результатов>
```

Проверка передачи на своих снимках без модели флокул, с пустой заглушкой сегментации:

```bash
.venv/bin/sludge-micro check-pipeline --passport model/passport.json --weights weights/microorganisms_fasterrcnn.pt --input <каталог снимков> --pipeline .. --report <файл отчёта JSON>
```

Команда копирует `cv_module` во временный каталог, запускает `AnalysisPipeline` с экспортом модели и сравнивает его детекции и счётчики с собственным инференсом. Для вызова внутри процесса `sludge_micro.adapter.PavelDetector` оборачивает предиктор из `sludge_micro.passport.passport_predictor` и реализует `detect(image_bgr)` с записями `Detection` пайплайна.

## Тесты

```bash
.venv/bin/pytest
```

Тесты строят синтетические изображения, архивы разметки и маленькие модели и не требуют весов и данных. Тест `test_upstream_pipeline_tests_pass_in_this_environment` запускает также тесты пайплайна Павла из корня репозитория. Проверки паспорта сверяют `model/passport.json`, блок паспорта в этом README, файлы `model/pavel/` и настройки `configs/frcnn_bn_frozen.yaml`. Если веса лежат в `weights/`, тест дополнительно сверяет их с паспортом. Перед запуском тесты проверяют, что установленный пакет совпадает с `src/`; после изменения кода пакет переустанавливается третьей командой установки.

## Паспорт модели

<!-- generated:passport -->

Паспорт `model/passport.json`: запуск `frcnn_bn_frozen`, статус: исследовательский прототип.

| Параметр | Значение |
| --- | --- |
| Архитектура | `fasterrcnn_resnet50_fpn_v2` |
| Вход | .jpg, .jpeg, .bmp; ожидаемый размер 1280 x 960 |
| Цвет и диапазон | OpenCV BGR `uint8` переводится в RGB `float32` от 0 до 1 |
| Масштабирование | короткая сторона 960, длинная не больше 1280 |
| Постобработка детектора | оценка от 0,05, NMS IoU 0,50, не больше 100 детекций |
| Порог confidence | 0,20, выбран на validation |
| Веса | `microorganisms_fasterrcnn.pt`, 173510743 байт, SHA-256 `0355b1ca23ad3cc88483d06de751f81b64a6a648f6888a2ff4d22fbb95e09af9` |

Качество: F1 считается по объектам при IoU не ниже 0,50, macro F1 равен среднему F1 по классам. Это не доля верно распознанных снимков.

| Выборка | Изображений | Объектов | Игнорируемых областей | Порог | macro F1 |
| --- | --- | --- | --- | --- | --- |
| validation | 46 | 172 | 8 | 0,20 | 0,324 |
| test, однократная оценка | 45 | 162 | 25 | 0,20 | 0,304 |

| Label | Класс | Название | Validation: объектов: P / R / F1 | Test: объектов: P / R / F1 | Test: ошибка подсчёта на кадр, средняя абсолютная / смещение |
| --- | --- | --- | --- | --- | --- |
| 1 | `attached_ciliates` | Прикреплённые инфузории | 62: 0,154 / 0,226 / 0,183 | 55: 0,268 / 0,400 / 0,321 | 1,31 / 0,60 |
| 2 | `filamentous_bacteria` | Нитчатые бактерии | 53: 0,168 / 0,491 / 0,250 | 45: 0,122 / 0,400 / 0,187 | 2,38 / 2,29 |
| 3 | `rotifers` | Коловратки | 20: 0,458 / 0,550 / 0,500 | 16: 0,619 / 0,812 / 0,703 | 0,20 / 0,11 |
| 4 | `testate_amoebae` | Раковинные амёбы | 19: 0,429 / 0,947 / 0,590 | 23: 0,448 / 0,565 / 0,500 | 0,49 / 0,13 |
| 5 | `free_swimming_ciliates` | Свободноплавающие инфузории | 8: 0,100 / 0,250 / 0,143 | 8: 0,200 / 0,375 / 0,261 | 0,38 / 0,16 |
| 6 | `nematoda` | Нематоды | 3: 0,143 / 0,333 / 0,200 | 9: 0,000 / 0,000 / 0,000 | 0,22 / -0,18 |
| 7 | `gastrotrichs` | Брюхоресничные | 7: 0,308 / 0,571 / 0,400 | 6: 0,143 / 0,167 / 0,154 | 0,24 / 0,02 |

Проверка передачи в пайплайн Павла на кадрах validation: кадров 46, объектов 364; детекции пайплайна совпали с собственным инференсом на 46 кадрах, счётчики совпали с детекциями на 46 кадрах, ошибок классов 0.

<!-- /generated:passport -->

Блок построен командой `sludge-micro passport-readme --passport model/passport.json --readme README.md` из паспорта, а паспорт — командой `sludge-micro passport` из отчётов запуска `frcnn_bn_frozen`. Test использован один раз, для итоговой оценки этой модели; модели и пороги по нему не выбирались.

## Повторное обучение

Для обучения нужен внешний набор данных: архивы ZIP с микроскопическими снимками JPG или BMP 1280 x 960 и разметкой COCO с категориями семи целевых классов, `unknown`, `custom_microorganism`, `debris` и других классов из раздела `classes` рабочей конфигурации. Имена архивов и их роли перечислены в разделе `sources`, каталог архивов задаёт `paths.raw_dir`. Набор и реальные снимки в репозиторий не входят. Обучение начинается с весов COCO torchvision `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` (FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1), путь к ним задаёт `paths.pretrained_weights`; программа ничего не скачивает.

```bash
.venv/bin/sludge-micro audit --config configs/frcnn_bn_frozen.yaml
.venv/bin/sludge-micro prepare --config configs/frcnn_bn_frozen.yaml
.venv/bin/sludge-micro variants --config configs/frcnn_bn_frozen.yaml
.venv/bin/sludge-micro split --config configs/frcnn_bn_frozen.yaml
.venv/bin/sludge-micro preflight --config configs/frcnn_bn_frozen.yaml
.venv/bin/sludge-micro train --config configs/frcnn_bn_frozen.yaml --run-id <запуск>
.venv/bin/sludge-micro passport --config configs/frcnn_bn_frozen.yaml --run-id <запуск> --output <паспорт JSON>
```

`train` оценивает запуск на validation и выбирает порог по сетке; `evaluate --split test` выполняется один раз для зафиксированной модели. Групповое разбиение и выбор копий Student_6 в `configs/frcnn_bn_frozen.yaml` повторяют настройки поставляемой модели; SHA-256 манифеста разбиения записан в паспорте. Обучение на CPU занимает порядка часа и более.

## Ограничения

- Исследовательский прототип: модель обучена одним запуском на небольшом наборе, оценки по классам предварительные. F1 — мера совпадения рамок и классов с разметкой, а не процент верно распознанных снимков.
- Нематоды: на test модель не нашла ни одного размеченного объекта этого класса, см. таблицу классов паспорта.
- Нитчатые бактерии: модель даёт много рамок этого класса и систематически пересчитывает их, см. смещение подсчёта в таблице паспорта.
- Разметка неполна: часть детекций приходится на неразмеченные объекты, поэтому precision может быть занижен, а ошибка подсчёта включает пропуски разметки.
- Разбиение невременное, test содержит кадры тех же архивов, что train. Проверки на новых пробах, датах съёмки и микроскопах не было.
- Редкие классы (нематоды, брюхоресничные, свободноплавающие инфузории) представлены в validation и test единицами объектов.
- Пайплайн Павла включает в счётчики детекции в областях, которые разметка отмечает как неопределённые; оценка их не учитывает.
- Зависимости, в том числе PyTorch, torchvision и Ultralytics, распространяются на своих лицензиях; исходные веса COCO получены из torchvision. Лицензия этого каталога не назначена.
