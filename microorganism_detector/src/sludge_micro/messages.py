"""User-facing messages in Russian and the project exception type.

Every message states the cause and the required action. Other modules must not
build user-facing text themselves; they pass a message key and values here.
"""

from __future__ import annotations

MESSAGES: dict[str, str] = {
    "file_missing": "Причина: не найден файл {path}. Действие: {action}",
    "config_invalid": (
        "Причина: конфигурация {path} не прошла проверку схемы: {details}. "
        "Действие: исправьте указанные поля."
    ),
    "project_root_missing": (
        "Причина: над файлом {path} не найден pyproject.toml. "
        "Действие: храните конфигурацию внутри каталога проекта."
    ),
    "archive_unsafe_member": (
        "Причина: элемент архива {member} выходит за целевой каталог. "
        "Действие: проверьте архив {archive}, извлечение остановлено."
    ),
    "archive_conflict": (
        "Причина: файл {path} уже существует и отличается от элемента архива. "
        "Действие: удалите производную копию или выберите другой каталог."
    ),
    "category_schema_mismatch": (
        "Причина: категория {name} в {source} имеет id {found}, в конфигурации {expected}. "
        "Действие: сверьте карту классов в configs/microorganisms.yaml с разметкой."
    ),
    "category_missing": (
        "Причина: категория {name} отсутствует в разметке {source}. "
        "Действие: проверьте архив или карту классов."
    ),
    "unknown_category": (
        "Причина: категория {name} из {source} не отнесена ни к одной группе классов. "
        "Действие: добавьте её в раздел classes конфигурации."
    ),
    "image_decode_failed": (
        "Причина: не удалось декодировать изображение {path}. "
        "Действие: проверьте файл или исключите его из набора."
    ),
    "unsupported_image": (
        "Причина: формат {suffix} не поддерживается, ожидается JPG или BMP. "
        "Действие: передайте изображение JPG или BMP."
    ),
    "no_images": "Причина: в {path} нет изображений JPG или BMP. Действие: укажите другой путь.",
    "image_id_collision": (
        "Причина: файлы {first} и {second} дают одинаковый image_id {image_id}. "
        "Действие: переименуйте один из файлов."
    ),
    "inventory_missing": (
        "Причина: отсутствует опись {path}. Действие: выполните sludge-micro audit."
    ),
    "prepared_missing": (
        "Причина: отсутствуют подготовленные данные {path}. "
        "Действие: выполните sludge-micro prepare."
    ),
    "temporal_metadata_missing": (
        "Причина: подтверждённое время съёмки есть у {dated} из {total} изображений, "
        "различных дат съёмки: {days}. Временное разбиение невозможно. "
        "Действие: предоставьте метаданные съёмки через split.owner_metadata "
        "или письменное решение владельца о другом протоколе."
    ),
    "temporal_boundaries_missing": (
        "Причина: не заданы границы split.train_end и split.validation_end. "
        "Действие: задайте границы до построения разбиения."
    ),
    "split_empty": (
        "Причина: выборка {split} пуста при заданных границах. "
        "Действие: пересмотрите границы разбиения."
    ),
    "split_leak": (
        "Причина: выборки {first} и {second} пересекаются по {key}: {count}. "
        "Действие: исправьте группировку перед обучением."
    ),
    "split_lock_mismatch": (
        "Причина: контрольная сумма {path} не совпадает с зафиксированной. "
        "Действие: восстановите разбиение или зафиксируйте новое до обучения."
    ),
    "pretrained_missing": (
        "Причина: не заданы или отсутствуют локальные исходные веса {path}. "
        "Действие: положите файл весов COCO для {architecture} в outputs/pretrained/ "
        "и укажите путь в paths.pretrained_weights. Программа веса не скачивает."
    ),
    "pretrained_hash_mismatch": (
        "Причина: содержимое {path} не совпадает с хешем {expected} в имени файла. "
        "Действие: замените файл копией, полученной из надёжного источника."
    ),
    "ultralytics_missing": (
        "Причина: пакет ultralytics не установлен в окружении проекта. "
        "Действие: установите его отдельной подготовкой окружения с разрешения владельца "
        "и зафиксируйте версию в requirements.lock."
    ),
    "yolo_rotation_unsupported": (
        "Причина: Ultralytics не поддерживает повороты на прямой угол вместе с bbox. "
        "Действие: отключите train.augment.rotation90 для кандидата YOLO."
    ),
    "checkpoint_unreadable": (
        "Причина: checkpoint {path} не читается: {details}. "
        "Действие: проверьте файл и его происхождение."
    ),
    "checkpoint_head_mismatch": (
        "Причина: голова checkpoint {path} рассчитана на {found} выходов, "
        "карта классов требует {expected}. Действие: используйте совместимые веса."
    ),
    "checkpoint_architecture_mismatch": (
        "Причина: checkpoint {path} несовместим с {architecture}: {details}. "
        "Действие: проверьте архитектуру и источник весов."
    ),
    "checkpoint_classes_missing": (
        "Причина: в checkpoint {path} нет подтверждённого порядка классов. "
        "Действие: запросите карту классов у автора весов."
    ),
    "checkpoint_classes_mismatch": (
        "Причина: порядок классов checkpoint {path} {found} отличается от конфигурации "
        "{expected}. Действие: используйте веса, обученные с текущей картой классов."
    ),
    "budget_exhausted": (
        "Причина: выполнено запусков {runs}, бюджет протокола {limit}. "
        "Действие: пересмотрите бюджет в разделе protocol до новых запусков."
    ),
    "resume_missing": (
        "Причина: нет сохранённого состояния для возобновления {path}. "
        "Действие: запустите обучение без --resume с новым идентификатором запуска."
    ),
    "compare_control_missing": (
        "Причина: контроль сравнения не определён однозначно, protocol.control_run: {run_id}. "
        "Действие: укажите в protocol.control_run идентификатор одного обученного контроля."
    ),
    "test_run_mismatch": (
        "Причина: запуск {run_id} не совпадает со своим манифестом по проверкам: {checks}. "
        "Действие: оцените на test запуск с весами и конфигурацией, записанными в манифесте."
    ),
    "split_owned_elsewhere": (
        "Причина: конфигурация читает зафиксированное разбиение из {path}, которым владеет "
        "другой этап. Действие: стройте разбиение конфигурацией этого этапа."
    ),
    "run_exists": (
        "Причина: запуск {run_id} уже содержит результаты в {path}. "
        "Действие: выберите другой идентификатор или возобновите запуск флагом --resume."
    ),
    "confidence_unselected": (
        "Причина: порог confidence для запуска {run_id} не выбран на validation. "
        "Действие: выполните sludge-micro evaluate --split validation."
    ),
    "test_already_used": (
        "Причина: тестовая выборка уже использована, запись {path}. "
        "Действие: для отчёта используйте сохранённые предсказания финальной оценки."
    ),
    "run_missing": ("Причина: не найден запуск {run_id} в {path}. Действие: выполните обучение."),
    "yolo_batchnorm_unsupported": (
        "Причина: режим BatchNorm {mode} задан для YOLO, а Ultralytics обучает BatchNorm сам. "
        "Действие: оставьте train.batchnorm: batch в конфигурации YOLO."
    ),
    "invalid_prediction": (
        "Причина: модель вернула недопустимый объект {details}. "
        "Действие: проверьте checkpoint и постобработку."
    ),
    "pavel_missing": (
        "Причина: не установлен пакет cv_module пайплайна Павла. Действие: в каталоге "
        "microorganism_detector выполните .venv/bin/pip install --no-deps --no-build-isolation .."
    ),
    "input_contract": (
        "Причина: вход не соответствует контракту {details}. "
        "Действие: передайте изображение uint8 с тремя каналами BGR."
    ),
    "network_disabled": (
        "Причина: программа попыталась открыть сетевое соединение {target}. "
        "Действие: предоставьте ресурс локально."
    ),
    "passport_invalid": (
        "Причина: паспорт модели {path} не соответствует схеме: {details}. "
        "Действие: используйте паспорт, записанный командой sludge-micro passport."
    ),
    "weights_mismatch": (
        "Причина: веса {path} не совпадают с паспортом, SHA-256 {found} вместо {expected}. "
        "Действие: загрузите веса выпуска, указанного в паспорте, и проверьте контрольную сумму."
    ),
    "passport_checkpoint_mismatch": (
        "Причина: checkpoint {path} расходится с паспортом по полям: {fields}. "
        "Действие: используйте веса и паспорт одного запуска."
    ),
    "weights_missing": (
        "Причина: для команды {command} указан --passport без --weights. "
        "Действие: укажите файл весов модели."
    ),
    "run_id_missing": (
        "Причина: для команды {command} не указан ни --passport, ни --run-id. "
        "Действие: укажите паспорт и веса поставляемой модели или запуск конфигурации."
    ),
    "readme_block_missing": (
        "Причина: в {path} нет блока <!-- generated:passport -->. "
        "Действие: добавьте пару маркеров блока паспорта в README."
    ),
    "pavel_pipeline_missing": (
        "Причина: в каталоге пайплайна Павла {path} нет пакета cv_module. "
        "Действие: укажите каталог, где лежит cv_module, параметром --pipeline или "
        "paths.pavel_pipeline."
    ),
    "pavel_patch_required": (
        "Причина: архитектура {architecture} требует изменённого пайплайна Павла. "
        "Действие: проверяйте передачу архитектур, которые пайплайн поддерживает без изменений."
    ),
}

STATUS: dict[str, str] = {
    "audit_done": "Аудит завершён, отчёт: {path}",
    "prepare_done": "Подготовка завершена, изображений в наборе: {included}, исключено: {excluded}",
    "split_done": "Разбиение записано: {path}",
    "variants_done": (
        "Вариантов обработки копий: {count}; рабочий вариант совпадает с подготовкой: {matches}"
    ),
    "proposal_done": (
        "Предложение границ записано: {path}. Перенесите границы в split после согласования."
    ),
    "preflight_done": "Проверка перед обучением записана: {path}",
    "train_done": "Обучение {run_id} завершено, checkpoint: {path}",
    "evaluate_done": "Оценка {run_id} на {split}: macro F1 {macro_f1}",
    "predict_done": "Предсказания записаны в {path}, изображений: {images}",
    "export_done": "Файлы для пайплайна Павла записаны в {path}",
    "compare_done": "Сравнение записано, запусков: {count}; выбранный запуск: {selected}",
    "examples_done": "Примеры запуска {run_id} записаны: {count}",
    "handover_done": (
        "Передача запуска {run_id} проверена на {images} кадрах; совпадений детекций: {matches}"
    ),
    "inspect_done": "Сведения о checkpoint записаны: {path}",
    "passport_done": "Паспорт модели записан: {path}",
    "model_ok": "Веса соответствуют паспорту запуска {run_id}; SHA-256 {sha256}",
    "pipeline_check_done": (
        "Проверка пайплайна записана: {path}; кадров: {images}; совпадений детекций: {matches}; "
        "совпадений счётчиков: {counts}"
    ),
    "readme_done": "Блок паспорта в {path} обновлён: {changed}",
}


def text(key: str, **values: object) -> str:
    """Format a user-facing message.

    Args:
        key: Message key from ``MESSAGES`` or ``STATUS``.
        **values: Values substituted into the template.

    Returns:
        The formatted Russian message.

    Raises:
        KeyError: If the key is unknown.
    """
    template = MESSAGES.get(key) or STATUS[key]
    return template.format(**values)


class ProjectError(Exception):
    """Error with a Russian message that names its cause and required action."""

    def __init__(self, key: str, **values: object) -> None:
        """Create the error from a message key.

        Args:
            key: Message key from ``MESSAGES``.
            **values: Values substituted into the template.
        """
        self.key = key
        self.values = values
        super().__init__(text(key, **values))
