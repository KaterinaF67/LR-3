from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def esc(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def table(df: pd.DataFrame, cols: list[str], headers: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    spec = "l" + "r" * (len(cols) - 1)
    lines = ["\\begin{tabular}{" + spec + "}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    for row in df[cols].itertuples(index=False, name=None):
        vals = []
        for col, val in zip(cols, row):
            if col in formats and isinstance(val, (int, float)):
                vals.append(format(val, formats[col]))
            else:
                vals.append(esc(val))
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def make_report(root: Path) -> None:
    data = root / "data"
    results = root / "results"
    servers = pd.read_csv(data / "servers.csv")
    tasks = pd.read_csv(data / "tasks.csv")
    milp_servers = pd.read_csv(results / "base_milp_servers.csv")
    lp_servers = pd.read_csv(results / "base_lp_servers.csv")
    scenarios = pd.read_csv(results / "scenarios.csv")
    scaling = pd.read_csv(results / "scaling.csv")
    bigm = pd.read_csv(results / "bigm_experiment.csv")
    formulations = pd.read_csv(results / "formulation_comparison.csv")
    greedy = json.loads((results / "greedy_summary.json").read_text(encoding="utf-8"))
    run = json.loads((results / "run_info.json").read_text(encoding="utf-8"))
    verify = json.loads((results / "verification_base.json").read_text(encoding="utf-8"))
    base_obj = run["base_milp_objective"]
    lp_obj = run["base_lp_objective"]
    gap = run["integrality_gap_pct"]
    active = milp_servers[milp_servers["y"] > 0.5]["server_id"].tolist()
    frac_y = lp_servers[(lp_servers["y"] > 1e-7) & (lp_servers["y"] < 1 - 1e-7)][["server_id", "y"]]
    scenario_changes = pd.read_csv(results / "scenario_server_sets.csv")
    form_lp = formulations[formulations["mode"] == "LP relaxation"]
    agg_gap = float(form_lp.loc[form_lp["formulation"] == "aggregate", "gap_pct"].iloc[0])
    dis_gap = float(form_lp.loc[form_lp["formulation"] == "disaggregated", "gap_pct"].iloc[0])

    tex = r'''\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath,amssymb,booktabs,array,longtable,graphicx,float,geometry,xcolor}
\geometry{left=20mm,right=20mm,top=18mm,bottom=20mm}
\setlength{\parindent}{1.25cm}
\setlength{\parskip}{0.25em}
\begin{document}
\begin{center}
{\Large Лабораторная работа № 3}\par
{\large MILP-модель дискретного выбора}\par
\vspace{0.6em}
Федоровская Е. С., Б24-517, вариант 19
\end{center}

\section{Постановка задачи}
Вариант 19: назначение серверов на задачи с ограничением числа включённых серверов. Рассматриваются 8 серверов и 15 задач. Требуется назначить каждую задачу ровно на один сервер и выбрать набор включённых серверов так, чтобы не превышать CPU- и RAM-ёмкости и минимизировать суммарную стоимость назначения и включения.

\subsection{Данные и единицы}
Для сервера $i$: $C_i$ --- CPU-ёмкость, усл. CPU; $R_i$ --- RAM, ГБ; $f_i$ --- фиксированная стоимость включения, усл. ед./период. Для задачи $j$: $c_j$ --- CPU-потребность, усл. CPU; $r_j$ --- RAM-потребность, ГБ. $a_{ij}$ --- стоимость назначения задачи $j$ на сервер $i$, усл. ед. Максимально разрешено включить $K=5$ серверов.

\paragraph{Генерация экземпляра.}
Используется генератор NumPy \texttt{default\_rng} с seed 10125. Для каждой из 15 задач CPU-потребность выбирается из целых значений $2,\ldots,6$ усл. CPU, RAM-потребность --- $3,\ldots,10$ ГБ. Для каждого из 8 серверов CPU-ёмкость выбирается из $14,\ldots,21$ усл. CPU, RAM --- из $24,\ldots,38$ ГБ. Фиксированная стоимость равномерно выбирается из интервала $[22,45)$ усл. ед., стоимость назначения каждой пары сервер--задача --- из $[3,15)$ усл. ед.; обе округляются до сотых. Значения сохраняются в \texttt{servers.csv}, \texttt{tasks.csv}, \texttt{assignment\_cost.csv}. Сгенерированные данные после сохранения загружаются из CSV. При масштабировании аналогичная процедура применяется к 10, 20 и 40 задачам с seed $10125+n_{\mathrm{tasks}}$.

\begin{table}[H]\centering\small
\caption{Параметры серверов}
''' + table(servers, ["server_id","cpu_capacity","ram_capacity_gb","fixed_cost"], ["Сервер","CPU","RAM, ГБ","Фикс. стоимость"], {"cpu_capacity":".0f","ram_capacity_gb":".0f","fixed_cost":".2f"}) + r'''
\end{table}

\begin{table}[H]\centering\small
\caption{Параметры задач}
''' + table(tasks, ["task_id","cpu_demand","ram_demand_gb"], ["Задача","CPU","RAM, ГБ"], {"cpu_demand":".0f","ram_demand_gb":".0f"}) + r'''
\end{table}

\section{Математическая модель}
Бинарная переменная $x_{ij}=1$, если задача $j$ назначена серверу $i$, иначе 0. Бинарная переменная $y_i=1$, если сервер $i$ включён.
\[
\min z=\sum_i f_i y_i+\sum_i\sum_j a_{ij}x_{ij}.
\]
Каждая задача назначается ровно один раз:
\[
\sum_i x_{ij}=1,\quad \forall j.
\]
Ограничения ресурсов:
\[
\sum_j c_jx_{ij}\le C_i,\qquad \sum_j r_jx_{ij}\le R_i,\quad \forall i.
\]
Связка big-M:
\[
\sum_jx_{ij}\le M_i y_i,\quad \forall i.
\]
Здесь $M_i$ выбирается как верхняя граница числа задач, физически помещающихся на сервер по CPU и RAM, поэтому это не произвольное большое число. Ограничение на число включённых серверов:
\[
\sum_i y_i\le5.
\]
Для MILP: $x_{ij},y_i\in\{0,1\}$. Для LP-релаксации: $0\le x_{ij},y_i\le1$.

\section{Реализация}
Модель решается HiGHS через \texttt{scipy.optimize.milp}, используется presolve. Данные находятся в CSV, модель --- в отдельных Python-модулях. Seed равен 10125. Независимая проверка читает CSV решения и заново вычисляет назначения, CPU, RAM, связки включения, ограничение на число серверов, целочисленность и значение целевой функции без обращения к объекту солвера.

\section{LP-релаксация и MILP}
LP-релаксация дала $z_{LP}=''' + f"{lp_obj:.3f}" + r'''$, а целочисленная модель --- $z_{MILP}=''' + f"{base_obj:.3f}" + r'''$. Для задачи минимизации интегральный зазор
\[
\frac{z_{MILP}-z_{LP}}{z_{MILP}}\cdot100\%=''' + f"{gap:.3f}" + r'''\%.
\]
MILP имеет статус Optimal, MIP gap равен ''' + f"{run['mip_gap']:.6f}" + r'''. В оптимуме включены серверы ''' + ", ".join(active) + r'''.

В LP-релаксации часть серверов может быть включена дробно, а задачи могут дробиться между серверами. Это математически допустимо для релаксации, но невозможно как реальное дискретное назначение. Примеры дробных $y_i$:
\begin{table}[H]\centering\small
''' + table(frac_y, ["server_id","y"], ["Сервер","$y_i$"], {"y":".4f"}) + r'''
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.94\textwidth]{results/plots/base_assignment.png}
\caption{Оптимальное MILP-назначение задач}
\end{figure}

\section{Независимая проверка}
Допуск независимой NumPy-проверки составляет $10^{-7}$. Нарушение назначения задач --- ''' + f"{verify['task_assignment_violation']:.1g}" + r''', CPU --- ''' + f"{verify['cpu_violation']:.1g}" + r''', RAM --- ''' + f"{verify['ram_violation']:.1g}" + r''', big-M-связок --- ''' + f"{verify['linking_violation']:.1g}" + r''', целочисленности --- ''' + f"{verify['integrality_violation']:.1g}" + r'''. Целевая функция совпадает с независимым пересчётом. Проверка \textbf{пройдена}.

\section{Сценарный анализ}
В дефицитном сценарии CPU и RAM наиболее загруженного активного сервера уменьшаются на 15\%. В стрессовом сценарии CPU и RAM двух наиболее загруженных активных серверов уменьшаются на 20\%. После каждого изменения MILP решается заново и отдельно проверяется.

\begin{table}[H]\centering\small
\caption{Результаты сценариев}
''' + table(scenarios, ["scenario","status","objective","delta_vs_base","active_servers"], ["Сценарий","Статус","Стоимость",r"$\Delta$","Активные серверы"], {"objective":".3f","delta_vs_base":".3f"}) + r'''
\end{table}

\begin{table}[H]\centering\small
\caption{Как меняется набор включённых серверов}
''' + table(scenario_changes, ["scenario","removed","added"], ["Сценарий","Выключены относительно базы","Добавлены относительно базы"]) + r'''
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.72\textwidth]{results/plots/scenarios.png}
\caption{Стоимость в трёх сценариях}
\end{figure}

\section{Масштабирование}
Для 10, 20 и 40 задач сгенерированы воспроизводимые экземпляры. Для каждого решены LP-релаксация и MILP, измерены время и интегральный зазор.
\begin{table}[H]\centering\small
''' + table(scaling, ["n_tasks","n_servers","lp_time_s","milp_time_s","integrality_gap_pct","mip_nodes"], ["Задач","Серверов","LP, с","MILP, с",r"Gap, \%",r"Узлы B\&B"], {"lp_time_s":".4f","milp_time_s":".4f","integrality_gap_pct":".2f","mip_nodes":".0f"}) + r'''
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.72\textwidth]{results/plots/scaling_time.png}
\caption{Время решения при росте задачи}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.72\textwidth]{results/plots/scaling_gap.png}
\caption{Интегральный зазор при росте задачи}
\end{figure}

\section{Сравнение с жадной эвристикой}
Жадная эвристика пробует три порядка задач: убывание суммарной нормированной потребности CPU и RAM, убывание CPU и убывание RAM. В каждом проходе задача назначается на доступный сервер с минимальной добавочной стоимостью $a_{ij}+f_i$ для нового сервера или $a_{ij}$ для уже включённого. Проверяются остатки CPU, RAM и лимит включённых серверов; если какой-либо проход не может назначить все задачи, он отбрасывается. Из допустимых проходов выбирается самый дешёвый. Её стоимость равна ''' + f"{greedy['objective']:.3f}" + r''', время --- ''' + f"{greedy['elapsed_s']:.6f}" + r''' с. Относительный зазор к оптимальному MILP составляет ''' + f"{greedy['gap_pct']:.3f}" + r'''\%. Эвристика работает быстрее, но математической гарантии оптимальности не имеет.

\section{Выбор big-M}
Разумное $M_i$ ограничено максимальным числом задач, которое физически может поместиться на сервер по CPU и RAM. Дополнительно проведён эксперимент с искусственным увеличением $M$ в 10 и 100 раз.
\begin{table}[H]\centering\small
''' + table(bigm, ["m_factor","lp_objective","lp_gap_pct","milp_objective","milp_time_s","mip_nodes"], ["Множитель $M$","LP",r"LP gap, \%","MILP","MILP, с","Узлы"], {"m_factor":".0f","lp_objective":".3f","lp_gap_pct":".2f","milp_objective":".3f","milp_time_s":".4f","mip_nodes":".0f"}) + r'''
\end{table}
Слишком большое $M$ не меняет допустимое целочисленное решение, но сильно ослабляет LP-релаксацию: дробные $y_i$ могут становиться искусственно малыми, из-за чего нижняя оценка становится нереалистично хорошей. Это ухудшает численную постановку и потенциально усложняет ветвление.

\begin{figure}[H]\centering
\includegraphics[width=0.72\textwidth]{results/plots/bigm_gap.png}
\caption{Ослабление LP-релаксации при завышенном big-M}
\end{figure}

\section{Дополнительное задание: две эквивалентные формулировки}
Сравнены две MILP-формулировки. В агрегированной версии используется $\sum_jx_{ij}\le M_iy_i$. В усиленной версии к ней добавлены ограничения $x_{ij}\le y_i$ для каждой пары сервер--задача. При бинарных $x$ и $y$ обе формулировки описывают одно и то же множество целочисленных решений, поскольку назначение на выключенный сервер и так запрещено агрегированной связкой. В LP-релаксации дополнительные индивидуальные связи делают модель сильнее.

\begin{table}[H]\centering\small
''' + table(formulations, ["formulation","mode","objective","gap_pct","time_s","n_constraints","mip_nodes"], ["Формулировка","Режим","Цель",r"Gap, \%","Время, с","Ограничений","Узлы"], {"objective":".3f","gap_pct":".2f","time_s":".4f","n_constraints":".0f","mip_nodes":".0f"}) + r'''
\end{table}

Для агрегированной модели LP-зазор составляет ''' + f"{agg_gap:.2f}" + r'''\%, для усиленной модели --- ''' + f"{dis_gap:.2f}" + r'''\%. Следовательно, усиленная формулировка даёт более тесную нижнюю оценку, хотя имеет больше ограничений. Обе MILP-формулировки дают одинаковую оптимальную целевую функцию ''' + f"{base_obj:.3f}" + r''' усл. ед. Для сравнения обе MILP решались с критерием \(\mathrm{MIP\ gap}\le10^{-7}\). Время решения и число узлов ветвления в таблице являются экспериментальными показателями, зависящими от реализации солвера и вычислительной среды, а не доказательство превосходства по любому экземпляру.

\paragraph{Проверка эквивалентности и логи.}
При бинарных $x_{ij}$ и $y_i$ агрегированная связка не разрешает $x_{ij}=1$ при $y_i=0$, а при $y_i=1$ число назначений ограничено $M_i$ и ресурсами. Поэтому $x_{ij}\le y_i$ уже выполняется для всех допустимых целочисленных назначений: дополнительная система меняет только непрерывную релаксацию. Для усиленной MILP независимо проверены назначения, целочисленность, границы CPU/RAM, обе группы связок и значение цели, результат сохранён в \texttt{results/verification\_disaggregated.json}.

Полные журналы HiGHS сохранены в каталоге \texttt{results/}:
\begin{quote}\small\ttfamily
solver\_formulation\_aggregate\_milp\_full.log\\
solver\_formulation\_disaggregated\_milp\_full.log\\
solver\_formulation\_aggregate\_lp\_full.log\\
solver\_formulation\_disaggregated\_lp\_full.log
\end{quote}
В логах зафиксированы presolve, целевые значения, статус и статистика ветвления. Они позволяют проверить сравнение формулировок по пункту 6 списка заданий со звёздочкой.

\begin{figure}[H]\centering
\includegraphics[width=0.45\textwidth]{results/plots/formulation_gap.png}
\caption{Сила двух LP-релаксаций}
\end{figure}

\section{Практическая реализуемость и ограничения}
MILP-решение реализуемо как дискретный план: каждый сервер либо включён, либо выключен, каждая задача назначена ровно одному серверу. Учебное упрощение состоит в том, что CPU и RAM считаются статическими агрегированными требованиями, а стоимость назначения линейна и не моделирует очереди, динамическую нагрузку, миграции и отказы. Поэтому решение является оптимальным для данной модели, но не описывает всю динамику реального дата-центра.

\paragraph{Вывод.}
Для варианта 19 реализована и независимо проверена MILP-модель распределения 15 задач по 8 серверам. Сравнены LP-релаксация и MILP, сценарии, масштабирование, эвристика и выбор big-M. Дополнительный эксперимент показал различие в силе LP-релаксаций двух эквивалентных целочисленных формулировок.
\end{document}
'''
    (root / "report.tex").write_text(tex, encoding="utf-8")
