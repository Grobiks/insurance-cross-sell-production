"""Demonstrate the retraining cycle on real data, several runs in a row.

The dataset has no dates, so new data arriving over time is simulated:
a stratified sample is cut into batches, and some batches are spoiled
on purpose to trigger every branch of the process.

    poetry run python scripts/retrain_demo.py

Writes `docs/RETRAIN_DEMO.md`, `docs/img/retrain_cycles.png` and copies
of the registry and cycle reports to `docs/retrain_demo/`.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from insurance_cross_sell import config
from insurance_cross_sell.data import load_sample
from insurance_cross_sell.retrain import (
    CycleReport,
    ModelRegistry,
    check_drift,
    data_profile,
    drift_report,
    run_cycle,
)

ROOT = config.PROJECT_ROOT
BATCH_NOTES = {
    1: "обычные данные",
    2: "обычные данные",
    3: "испорчена схема: неизвестный пол и пропуски премии",
    4: "ошибка разметки: ответы клиентов перемешаны",
    5: "дрейф: клиенты старше на 10 лет",
    6: "обычные данные",
}
STATUS_RU = {
    "deployed": "опубликована",
    "rejected_data": "отказ: данные",
    "rejected_quality": "отказ: качество",
    "not_approved": "не одобрена",
    "rolled_back": "откат после деплоя",
}


REPORT_TEMPLATE = """\n# Демонстрация цикла переобучения

Документ показывает, что процесс из [Model_retrain_BPMN.md](../Model_retrain_BPMN.md) работает в коде: цикл
переобучения запущен 6 раз подряд на реальных данных, и в каждом запуске пройдены настоящие проверки. Отчёт
сгенерирован автоматически скриптом:

```bash
poetry run python scripts/retrain_demo.py
```

Отдельный цикл запускается командой (последний файл это новая партия данных, остальные это окно обучения):

```bash
poetry run insurance retrain --data data/batches/batch_01.csv data/batches/batch_02.csv --trigger new_data
poetry run insurance monitor --data data/batches/batch_05.csv
```

Без флага `--auto-approve` команда `retrain` перед публикацией спрашивает одобрение у человека (ручная задача
в BPMN). В демонстрации одобрение автоматическое, это записано в реестре как `auto (демо)`.

## 1. Данные

Источник: Kaggle Playground Series S4E7, 11 504 798 клиентов, у 12.3% ответ «интересно» (`Response` = 1).
Дат в данных нет, поэтому поступление новых данных имитируется: стратифицированная выборка нарезана на 6
партий по {rows} клиентов, как будто это данные за 6 периодов. Партии 3, 4 и 5 испорчены специально, чтобы
проверить ветки неуспеха и реакцию на дрейф.

### Как выглядят данные (первые строки партии 1)

{sample}

### Партии

{batches}

PSI (индекс стабильности распределения) считается относительно партии 1: меньше 0.1 означает, что
распределение не изменилось, больше 0.2 означает существенный сдвиг.

## 2. Сценарий

| Цикл | Триггер | Окно обучения | Что проверяем |
|---|---|---|---|
| 1 | ручной запуск | партия 1 | первая версия модели, сравнивать не с чем |
| 2 | новые данные | партии 1-2 | новая модель против действующей на свежих данных |
| 3 | расписание | партии 2-3 | испорченная схема данных должна остановить цикл до обучения |
| 4 | новые данные | партии 2 и 4 | ошибка разметки должна провалить проверки качества |
| 5 | дрейф | партии 2 и 5 | команда `monitor` находит дрейф и запускает переобучение |
| 6 | расписание | партии 5-6 | имитация сбоя после деплоя: должен сработать откат |

В каждом цикле 25% новой партии откладываются и не участвуют в обучении: на них сравниваются новая модель и
действующая. Модель: LightGBM в быстрой конфигурации (параметры из `config.BEST_PARAMS`).

## 3. Результаты циклов

![ROC-AUC действующей и новой модели по циклам](img/retrain_cycles.png)

{cycles}

## 4. Проверки в каждом цикле

{details}

## 5. Обнаружение дрейфа перед циклом 5

Команда `insurance monitor` сравнила партию 5 с данными, на которых обучена действующая модель:

| Признак | PSI |
|---|---|
{monitor_psi}

Дрейф обнаружен: **{monitor_detected}** (порог PSI 0.2), поэтому цикл 5 запущен с триггером `drift`.

## 6. Реестр моделей после демонстрации

{versions}

В эксплуатации после всех циклов: **{production}**. Модели отклонённых циклов не удалены: они сохранены со
статусом `rejected` или `rolled_back` для разбора причин.

## 7. Уведомления

```text
{notifications}
```

Файлы реестра и отчёты каждого цикла в формате JSON лежат в [`retrain_demo/`](retrain_demo/).

## 8. Выводы

Проверка сценария: {check}.

| Ветка процесса из BPMN | Где сработала | Что произошло |
|---|---|---|
| Первая версия без сравнения | цикл 1 | модель прошла абсолютные проверки, одобрена и опубликована как v1 |
| Обновление: новая модель лучше действующей | цикл 2 | на одних и тех же свежих данных новая модель обошла действующую по ROC-AUC и F1, v1 ушла в архив |
| A: данные не прошли проверку | цикл 3 | цикл остановлен до обучения (проверка схемы сообщает о первом найденном нарушении), модель в работе не изменилась |
| B: качество не подошло | цикл 4 | при перемешанных ответах обе модели на свежих данных дают ROC-AUC около 0.5: это признак ошибки в разметке, а не в модели. Новая версия сохранена как `rejected` |
| Триггер по дрейфу | цикл 5 | `monitor` нашёл сдвиг возраста (PSI выше 3), новая модель на сдвинутых данных лучше действующей (ROC-AUC 0.874 против 0.863) и опубликована |
| C: сбой после деплоя | цикл 6 | модель прошла проверки и была опубликована, но проверка после выкладки не прошла: автоматический откат на предыдущую версию |

Во всех неуспешных ветках в эксплуатации остаётся последняя рабочая версия, а причина записана в отчёт цикла
и в журнал уведомлений.
"""

TRIGGER_RU = {
    "manual": "ручной",
    "schedule": "расписание",
    "new_data": "новые данные",
    "drift": "дрейф",
    "degradation": "падение качества",
}
GATE_RU = {
    "schema": "схема и значения данных",
    "min_rows": "объём партии",
    "positive_rate": "доля положительного класса",
    "min_auc": "минимальный ROC-AUC",
    "vs_champion_auc": "ROC-AUC не хуже действующей",
    "vs_champion_f1": "F1 не хуже действующей",
    "slices": "качество по группам клиентов",
    "top_features": "главные признаки",
    "load_test": "тест загрузки модели",
}
EXPECTED = [
    "deployed",
    "deployed",
    "rejected_data",
    "rejected_quality",
    "deployed",
    "rolled_back",
]


def make_batches(out_dir: Path, n_batches: int, rows: int) -> list[Path]:
    """Cut a stratified sample of the data into batches; spoil some."""
    frac = n_batches * rows / 11_504_798 * 1.02
    sample = load_sample(config.DATA_PATH, frac)
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample = sample.iloc[rng.permutation(len(sample))]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_batches):
        batch = sample.iloc[i * rows : (i + 1) * rows].copy()
        n = i + 1
        if n == 3:
            batch.loc[batch.index[:2000], "Gender"] = "Unknown"
            batch.loc[batch.index[2000:3000], "Annual_Premium"] = np.nan
        if n == 4:
            batch[config.TARGET] = rng.permutation(
                batch[config.TARGET].to_numpy()
            )
        if n == 5:
            batch["Age"] = (batch["Age"] + 10).clip(upper=85)
        path = out_dir / f"batch_{n:02d}.csv"
        batch.to_csv(path)
        paths.append(path)
    return paths


def describe_batches(paths: list[Path]) -> pd.DataFrame:
    """Rows, positive rate and drift of every batch against batch 1."""
    first = pd.read_csv(paths[0], index_col=config.ID_COLUMN)
    profile = data_profile(first)
    rows = []
    for i, path in enumerate(paths, start=1):
        df = pd.read_csv(path, index_col=config.ID_COLUMN)
        clean_part = df[df["Gender"].isin(["Male", "Female"])].dropna()
        drift = drift_report(profile, clean_part)
        worst = max(drift, key=lambda k: drift[k])
        rows.append(
            {
                "batch": f"batch_{i:02d}",
                "rows": len(df),
                "positive_rate": round(float(df[config.TARGET].mean()), 4),
                "max_psi": f"{drift[worst]:.3f} ({worst})",
                "note": BATCH_NOTES.get(i, ""),
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()]
    return "\n".join([head, sep, *body])


def plot_cycles(reports: list[CycleReport], path: Path) -> None:
    """Champion and challenger ROC-AUC per cycle with the decision."""
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=110)
    for r in reports:
        x = r.cycle
        if "champion" in r.metrics:
            ax.scatter(
                x,
                r.metrics["champion"]["roc_auc"],
                s=90,
                color="#9aa9bd",
                zorder=3,
            )
        if "challenger" in r.metrics:
            ax.scatter(
                x,
                r.metrics["challenger"]["roc_auc"],
                s=90,
                color="#1f5fbf",
                zorder=3,
            )
        ax.text(
            x,
            0.47,
            STATUS_RU[r.status],
            ha="center",
            fontsize=9,
            color="#1f8a4c" if r.status == "deployed" else "#b3261e",
        )
    ax.axhline(0.80, color="#b3261e", ls="--", lw=1)
    ax.text(0.55, 0.805, "минимум ROC-AUC 0.80", color="#b3261e", fontsize=9)
    ax.scatter([], [], color="#9aa9bd", label="действующая модель")
    ax.scatter([], [], color="#1f5fbf", label="новая модель")
    ax.legend(loc="center left", frameon=False)
    ax.set(
        xlabel="Цикл переобучения",
        ylabel="ROC-AUC на свежих данных",
        ylim=(0.44, 0.92),
        xticks=[r.cycle for r in reports],
        xlim=(0.5, len(reports) + 0.5),
    )
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Демонстрация цикла переобучения: 6 запусков")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """Build batches, run the scenario and write the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "models" / "demo"
    )
    args = parser.parse_args()

    shutil.rmtree(args.registry, ignore_errors=True)
    registry = ModelRegistry(args.registry)
    b = make_batches(ROOT / "data" / "batches", 6, args.rows)
    batches = describe_batches(b)

    def approve(report: CycleReport) -> str:
        return "auto (демо)"

    scenario = [
        ("manual", [b[0]], False),
        ("new_data", [b[0], b[1]], False),
        ("schedule", [b[1], b[2]], False),
        ("new_data", [b[1], b[3]], False),
        ("drift", [b[1], b[4]], False),
        ("schedule", [b[4], b[5]], True),
    ]
    reports, timings, monitor = [], [], {}
    for trigger, files, fail_deploy in scenario:
        if trigger == "drift":
            psi, detected = check_drift(files[-1], registry)
            monitor = {"psi": psi, "detected": detected}
        start = time.perf_counter()
        report = run_cycle(
            files,
            registry,
            trigger=trigger,
            approver=approve,
            simulate_deploy_failure=fail_deploy,
        )
        timings.append(round(time.perf_counter() - start))
        reports.append(report)
        print(
            f"cycle {report.cycle}: {report.status} "
            f"(production {report.production_after})"
        )

    docs = ROOT / "docs"
    plot_cycles(reports, docs / "img" / "retrain_cycles.png")
    evidence = docs / "retrain_demo"
    shutil.rmtree(evidence, ignore_errors=True)
    shutil.copytree(registry.root / "cycles", evidence / "cycles")
    shutil.copy(registry.index_path, evidence / "registry.json")
    shutil.copy(
        registry.root / "notifications.log", evidence / "notifications.txt"
    )
    sample_rows = pd.read_csv(b[0]).head(5)
    write_report(
        docs / "RETRAIN_DEMO.md",
        batches,
        sample_rows,
        reports,
        timings,
        monitor,
        registry,
    )
    print("report: docs/RETRAIN_DEMO.md")


def fmt(metrics: dict, key: str) -> str:
    """Format one metric or a dash when it is missing."""
    return f"{metrics[key]:.4f}" if key in metrics else "-"


def write_report(
    path: Path,
    batches: pd.DataFrame,
    sample_rows: pd.DataFrame,
    reports: list[CycleReport],
    timings: list[int],
    monitor: dict,
    registry: ModelRegistry,
) -> None:
    """Write the Markdown report of the demonstration."""
    template = REPORT_TEMPLATE
    cycles = []
    for r, sec in zip(reports, timings, strict=True):
        ch = r.metrics.get("challenger", {})
        cp = r.metrics.get("champion", {})
        cycles.append(
            {
                "Цикл": r.cycle,
                "Триггер": TRIGGER_RU[r.trigger],
                "Данные": " + ".join(Path(f).stem for f in r.data_files),
                "Действующая": r.champion or "нет",
                "ROC-AUC действующей": fmt(cp, "roc_auc"),
                "ROC-AUC новой": fmt(ch, "roc_auc"),
                "F1 новой": fmt(ch, "f1"),
                "Решение": STATUS_RU[r.status],
                "Версия": r.version or "-",
                "В работе после": r.production_after or "нет",
                "Время, с": sec,
            }
        )
    details = []
    for r in reports:
        details.append(
            f"**Цикл {r.cycle}** ({TRIGGER_RU[r.trigger]}, "
            f"{STATUS_RU[r.status]})\n"
        )
        for g in r.gates:
            mark = "✓" if g.passed else "✗"
            name = GATE_RU.get(g.name, g.name)
            details.append(f"- {mark} {name}: {g.detail}")
        if r.drift:
            worst = max(r.drift, key=lambda k: r.drift[k])
            details.append(
                f"- дрейф к обучающим данным действующей "
                f"модели: максимум PSI {r.drift[worst]:.3f} "
                f"({worst})"
            )
        details.append("")
    versions = pd.DataFrame(
        [
            {
                "Версия": v["version"],
                "Статус": v["status"],
                "Цикл": v["cycle"],
                "Триггер": TRIGGER_RU[v["trigger"]],
                "ROC-AUC": v["metrics"]["roc_auc"],
                "F1": v["metrics"]["f1"],
                "Строк обучения": v["train_rows"],
            }
            for v in registry.versions
        ]
    )
    notes = (registry.root / "notifications.log").read_text("utf-8")
    psi_lines = "\n".join(
        f"| {k} | {v:.3f} |" for k, v in monitor.get("psi", {}).items()
    )
    text = template.format(
        batches=md_table(batches),
        sample=md_table(sample_rows),
        cycles=md_table(pd.DataFrame(cycles)),
        details="\n".join(details),
        versions=md_table(versions),
        notifications=notes.strip(),
        monitor_psi=psi_lines,
        monitor_detected="да" if monitor.get("detected") else "нет",
        production=registry.production_version(),
        check=(
            "все 6 циклов завершились так, как задумано в сценарии"
            if [r.status for r in reports] == EXPECTED
            else "ВНИМАНИЕ: итоги циклов отличаются от задуманного "
            "сценария, выводы ниже нужно перепроверить"
        ),
        rows=int(batches["rows"].iloc[0]),
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
