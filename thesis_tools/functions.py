from typing import Dict, List, Optional, Sequence, Tuple

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# Configuración global del renderizador para Plotly (evita repetirlo en cada función)
pio.renderers.default = "iframe"

# Paletas y estilos reutilizables (constantes de módulo; no cambian la API)
_PALETTE: List[str] = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]
_GROUP_COLOR_PAIRS: List[Tuple[str, str]] = [
    ("#636EFA", "#AB63FA"),
    ("#00CC96", "#19D3F3"),
    ("#EF553B", "#FFA15A"),
    ("#FF6692", "#B6E880"),
    ("#FF97FF", "#FECB52"),
]
_IDENTITY_COLORS: List[str] = ["#EF553B", "#FFA15A"]
_SCATTER_BASE_COLOR: str = "#0D2A63"


# -----------------------------------------------------------------------------
# 1) GRÁFICAS DE DISPERSIÓN
# -----------------------------------------------------------------------------

def scatter_plot(
    filepath: str,
    columns: Sequence[str],
    traces_colors: Optional[Sequence[str]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    legend_labels: Optional[Sequence[str]] = None,
):
    """
    Genera un scatter plot a partir de un archivo, utilizando las columnas 
    especificadas en función del índice temporal del conjunto de datos.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - columns (list): Lista de nombres de columnas a graficar en el eje Y.
    - traces_colors (list, opcional): Lista de colores en formato hexadecimal para cada traza.
    - x_label (str, opcional): Etiqueta para el eje X.
    - y_label (str, opcional): Etiqueta para el eje Y.
    - legend_labels (list, opcional): Lista de nombres personalizados para la leyenda.

    Retorna:
    - None: Muestra el scatter plot interactivo.
    """
    data = pd.read_parquet(filepath)

    cols = list(columns)
    legends = list(legend_labels) if legend_labels and len(legend_labels) == len(cols) else cols
    colors = list(traces_colors) if traces_colors and len(traces_colors) == len(cols) else _PALETTE[: len(cols)]

    fig = go.Figure()
    for col, color, legend in zip(cols, colors, legends):
        fig.add_trace(
            go.Scatter(x=data.index, y=data[col], mode="markers", marker=dict(color=color), name=legend)
        )

    fig.update_layout(
        xaxis_title=x_label or "Índice",
        yaxis_title=y_label or ", ".join(cols),
        hovermode="x unified",
    )
    return fig


def scatter_subplots(filepath: str, subplot_configs: Sequence[Dict]):
    """
    Genera subplots de gráficos scatter a partir de un archivo, permitiendo definir en cada subplot
    las columnas a graficar y sus respectivas opciones.
    
    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - subplot_configs (list of dict): Lista de diccionarios, donde cada diccionario contiene:
        - 'columns' (list): Lista de nombres de columnas a graficar en el subplot.
        - 'traces_colors' (list, opcional): Lista de colores en formato hexadecimal para cada traza.
        - 'x_label' (str, opcional): Etiqueta para el eje X.
        - 'y_label' (str, opcional): Etiqueta para el eje Y.
        - 'legend_labels' (list, opcional): Lista de nombres personalizados para la leyenda.
    
    Retorna:
    - None: Muestra los scatter subplots interactivos.
    """
    data = pd.read_parquet(filepath)

    num_subplots = len(subplot_configs)
    fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True)

    for i, cfg in enumerate(subplot_configs, start=1):
        cols = list(cfg.get("columns", []))
        colors = list(cfg.get("traces_colors", _PALETTE[: len(cols)]))
        legends = list(cfg.get("legend_labels", cols))
        x_label = cfg.get("x_label", "Índice")
        y_label = cfg.get("y_label", ", ".join(cols))

        for col, color, legend in zip(cols, colors, legends):
            fig.add_trace(
                go.Scatter(x=data.index, y=data[col], mode="markers", marker=dict(color=color), name=legend),
                row=i,
                col=1,
            )

        fig.update_yaxes(title_text=y_label, row=i, col=1)

    fig.update_xaxes(title_text=x_label, row=num_subplots, col=1)
    fig.update_layout(hovermode="x unified")
    return fig


# -----------------------------------------------------------------------------
# 2) GRÁFICAS XY (columna vs columna)
# -----------------------------------------------------------------------------

def columns_plot(
    filepath: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    trace_color: Optional[str] = None,
    legend_name: Optional[str] = None,
    identity_line: bool = False,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
):
    """
    Genera un scatter plot usando las columnas especificadas para los ejes X e Y.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - x_col (str): Nombre de la columna a graficar en el eje X.
    - y_col (str): Nombre de la columna a graficar en el eje Y.
    - x_label (str): Etiqueta para el eje X.
    - y_label (str): Etiqueta para el eje Y.
    - trace_color (str, opcional): Color de la traza en formato hexadecimal.
    - legend_name (str, opcional): Texto que se visualizará en la leyenda para la traza.
    - identity_line (bool, opcional): Si es True, añade una línea y=x con nombre "Pendiente (m=1)".
    - x_range (tuple de float, opcional): Rango manual para el eje X (min, max).
    - y_range (tuple de float, opcional): Rango manual para el eje Y (min, max).

    Retorna:
    - None: Muestra el scatter plot interactivo.
    """
    data = pd.read_parquet(filepath)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode="markers",
            marker=dict(color=trace_color or _PALETTE[0]),
            name=(legend_name if legend_name is not None else None),
        )
    )

    if identity_line:
        # Determinar rango apropiado para y=x considerando rangos manuales cuando existan
        x_min, x_max = (float(data[x_col].min()), float(data[x_col].max()))
        y_min, y_max = (float(data[y_col].min()), float(data[y_col].max()))
        if x_range:
            x_min, x_max = min(x_min, x_range[0]), max(x_max, x_range[1])
        if y_range:
            y_min, y_max = min(y_min, y_range[0]), max(y_max, y_range[1])
        lo, hi = min(x_min, y_min), max(x_max, y_max)

        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(dash="dash"),
                name="Pendiente (m=1)",
            )
        )
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[lo, hi])

    if x_range:
        fig.update_xaxes(range=list(x_range))
    if y_range:
        fig.update_yaxes(range=list(y_range))

    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, showlegend=True)
    return fig


def columns_subplots(
    filepath: str,
    subplot_configs: Sequence[Dict],
    identity_line: bool = False,
):
    """
    Genera subplots de gráficos scatter utilizando columnas específicas para los ejes X e Y.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - subplot_configs (list of dict): Lista de diccionarios, donde cada uno debe contener:
        - 'x_col' (str): Nombre de la columna para el eje X.
        - 'y_col' (str): Nombre de la columna para el eje Y.
        - 'x_label' (str): Etiqueta para el eje X.
        - 'y_label' (str): Etiqueta para el eje Y.
        - 'trace_color' (str, opcional): Color de la traza en formato hexadecimal.
        - 'legend_name' (str, opcional): Texto que se visualizará en la leyenda para la traza.
        - 'x_range' (list of float, opcional): Rango manual para el eje X [min, max].
        - 'y_range' (list of float, opcional): Rango manual para el eje Y [min, max].
    - identity_line (bool, opcional): Si es True, añade línea y=x con etiqueta "Pendiente (m=1)".

    Retorna:
    - None: Muestra los scatter subplots interactivos.
    """
    data = pd.read_parquet(filepath)

    n = len(subplot_configs)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False)

    for i, cfg in enumerate(subplot_configs, start=1):
        x = data[cfg["x_col"]]
        y = data[cfg["y_col"]]
        x_label = cfg.get("x_label", cfg["x_col"])  # por defecto, nombre de la columna
        y_label = cfg.get("y_label", cfg["y_col"])  # idem
        color = cfg.get("trace_color", _PALETTE[0])
        legend = cfg.get("legend_name", cfg["y_col"])

        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", marker=dict(color=color), name=legend),
            row=i,
            col=1,
        )

        fig.update_xaxes(title_text=x_label, row=i, col=1)
        fig.update_yaxes(title_text=y_label, row=i, col=1)

        x_range = cfg.get("x_range")
        y_range = cfg.get("y_range")
        if x_range:
            fig.update_xaxes(range=x_range, row=i, col=1)
        if y_range:
            fig.update_yaxes(range=y_range, row=i, col=1)

        if identity_line:
            # Determinar rango de la línea identidad en función de rangos visibles
            x_lo = x_range[0] if x_range else float(x.min())
            x_hi = x_range[1] if x_range else float(x.max())
            y_lo = y_range[0] if y_range else float(y.min())
            y_hi = y_range[1] if y_range else float(y.max())
            lo, hi = min(x_lo, y_lo), max(x_hi, y_hi)
            color_line = _IDENTITY_COLORS[(i - 1) % len(_IDENTITY_COLORS)]

            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    line=dict(dash="dash", color=color_line),
                    name="Pendiente (m=1)",
                ),
                row=i,
                col=1,
            )

    fig.update_layout(showlegend=True, height=400 * n)
    return fig


# -----------------------------------------------------------------------------
# 3) GRÁFICAS CON MENÚ DESPLEGABLE
# -----------------------------------------------------------------------------

def dropdowns_plot(
    filepath: str,
    trace_labels: Sequence[str],
    x_cols: Sequence[str],
    y_cols: Sequence[str],
    x_axis_title: str,
    y_axis_title: str,
):
    """
    Genera un gráfico scatter interactivo con un menú desplegable para alternar entre diferentes trazas.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - trace_labels (list of str): Lista con los labels de cada traza.
    - x_cols (list of str): Lista con el nombre de la columna para el eje X de cada traza.
    - y_cols (list of str): Lista con el nombre de la columna para el eje Y de cada traza.
    - x_axis_title (str): Título que se asignará al eje X en el layout.
    - y_axis_title (str): Título que se asignará al eje Y en el layout.

    Retorna:
    - None: Muestra los scatter plots interactivos.
    """
    data = pd.read_parquet(filepath)

    n = len(trace_labels)
    if not (len(x_cols) == len(y_cols) == n):
        raise ValueError("`trace_labels`, `x_cols` y `y_cols` deben tener la misma longitud.")

    fig = go.Figure()
    for i in range(n):
        fig.add_trace(
            go.Scatter(
                x=data[x_cols[i]],
                y=data[y_cols[i]],
                mode="markers",
                marker=dict(color=_PALETTE[i % len(_PALETTE)]),
                name=trace_labels[i],
                visible=(i == 0),
            )
        )

    buttons = []
    for i in range(n):
        visibility = [j == i for j in range(n)]
        buttons.append({"label": trace_labels[i], "method": "update", "args": [{"visible": visibility}]})

    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
    )
    return fig


def compare_plot(
    filepath: str,
    group_dict: Dict[str, Tuple[str, str]],
    x_label: str = "Tiempo",
    y_label: Optional[Dict[str, str]] = None,
    legend_labels: Optional[Dict[str, Tuple[str, str]]] = None,
):
    """
    Genera un gráfico scatter interactivo con un menú desplegable para alternar entre diferentes variables.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - group_dict (dict): Diccionario donde cada clave es la etiqueta del menú (ej. "Temperatura (°C)") y
      el valor es una tupla o lista con dos elementos: (columna_sensor, columna_referencia).
    - x_label (str): Título del eje X (por defecto "Tiempo").
    - y_label (dict, opcional): Diccionario con los títulos personalizados del eje Y para cada menú.
    - legend_labels (dict, opcional): Diccionario donde cada clave es una etiqueta del menú y su valor
      es una tupla con los nombres personalizados para la leyenda de las dos trazas.

    Retorna:
    - None: Muestra el gráfico interactivo.
    """
    data = pd.read_parquet(filepath)

    groups = list(group_dict.keys())
    traces_per_group = 2
    total_traces = len(groups) * traces_per_group

    fig = go.Figure()

    for i, group in enumerate(groups):
        sensor_col, ref_col = group_dict[group]
        sensor_color, ref_color = _GROUP_COLOR_PAIRS[i % len(_GROUP_COLOR_PAIRS)]

        sensor_legend = (
            legend_labels[group][0] if legend_labels and group in legend_labels else sensor_col
        )
        ref_legend = legend_labels[group][1] if legend_labels and group in legend_labels else ref_col

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[sensor_col],
                mode="markers",
                marker=dict(color=sensor_color),
                name=sensor_legend,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[ref_col],
                mode="markers",
                marker=dict(color=ref_color),
                name=ref_legend,
            )
        )

    # visibilidad inicial (primer grupo)
    for j in range(total_traces):
        fig.data[j].visible = j < traces_per_group

    if y_label is None:
        y_label = {g: g for g in groups}

    buttons = []
    for i, group in enumerate(groups):
        visibility = [False] * total_traces
        start = i * traces_per_group
        visibility[start] = True
        visibility[start + 1] = True
        buttons.append(
            {
                "label": group,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"yaxis": {"title": {"text": y_label.get(group, group)}}},
                ],
            }
        )

    fig.update_layout(
        hovermode="x unified",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        xaxis_title=x_label,
        yaxis_title=y_label.get(groups[0], groups[0]) if groups else None,
    )
    return fig


# -----------------------------------------------------------------------------
# 4) REGRESIONES LINEALES
# -----------------------------------------------------------------------------

def linear_reg_subplots(
    filepath: str,
    columns: Sequence[str],
    time_intervals: Sequence[Tuple[str, str, float]],
    titles: Sequence[str],
):
    """
    Genera subplots con gráficos de dispersión y líneas de regresión para cada columna.

    Parámetros:
        - filepath (str): Ruta al archivo Parquet.
        - columns (list): Lista de nombres de columnas para la regresión.
        - time_intervals (list): Lista de tuplas (inicio, fin, valor_Tref).
        - titles (list): Lista de títulos para cada subplot (debe coincidir en longitud con 'columns').

    Retorna:
    - None: Muestra los gráficos interactivos en subplots.
    """
    if len(columns) != len(titles):
        raise ValueError("La cantidad de títulos debe ser igual a la cantidad de columnas.")

    data = pd.read_parquet(filepath)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex.")

    df = data.copy()
    df["Tref"] = np.nan
    for start, end, tref_value in time_intervals:
        idx = df.between_time(start, end).index
        df.loc[idx, "Tref"] = tref_value

    df = df.dropna(subset=["Tref"])  # datos con referencia

    n = len(columns)
    n_rows = math.ceil(math.sqrt(n))
    n_cols = math.ceil(n / n_rows)

    fig = make_subplots(
        rows=n_rows, cols=n_cols, shared_xaxes=True, shared_yaxes=True, subplot_titles=list(titles)
    )

    for idx, col in enumerate(columns):
        tmp = df[[col, "Tref"]].dropna()
        if tmp.empty:
            continue
        X = tmp[col].to_numpy().reshape(-1, 1)
        y = tmp["Tref"].to_numpy()

        model = LinearRegression().fit(X, y)
        y_pred = (model.coef_[0] * X.ravel()) + model.intercept_

        row = idx // n_cols + 1
        col_subplot = idx % n_cols + 1

        fig.add_trace(
            go.Scatter(x=X.ravel(), y=y, mode="markers", marker=dict(color=_SCATTER_BASE_COLOR), name=f"Datos {col}"),
            row=row,
            col=col_subplot,
        )
        fig.add_trace(
            go.Scatter(
                x=X.ravel(),
                y=y_pred,
                mode="lines",
                line=dict(color=_PALETTE[idx % len(_PALETTE)], width=2),
                name=f"Ajuste {col}",
            ),
            row=row,
            col=col_subplot,
        )

    fig.update_xaxes(tickmode="linear", dtick=10)
    fig.update_yaxes(tickmode="linear", dtick=10)

    for j in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Temperatura de termopar (°C)", row=n_rows, col=j)
    for i in range(1, n_rows + 1):
        fig.update_yaxes(title_text="T<sub>ref</sub> (°C)", row=i, col=1)

    fig.update_layout(showlegend=False)
    return fig


def linear_reg_plot(
    filepath: str,
    columns: Sequence[str],
    time_intervals: Sequence[Tuple[str, str, float]],
):
    """
    Genera gráficos de dispersión con líneas de regresión para cada termopar.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - columns (list): Lista de nombres de columnas para la regresión.
    - time_intervals (list): Lista de tuplas (inicio, fin, valor_Tref).

    Retorna:
    - None: Muestra los gráficos interactivos en un dropdown.
    """
    data = pd.read_parquet(filepath)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex.")

    df = data.copy()
    df["Tref"] = np.nan
    for start, end, tref_value in time_intervals:
        idx = df.between_time(start, end).index
        df.loc[idx, "Tref"] = tref_value

    df = df.dropna(subset=["Tref"])  # datos con referencia

    fig = go.Figure()

    for i, col in enumerate(columns):
        tmp = df[[col, "Tref"]].dropna()
        X = tmp[col].to_numpy().reshape(-1, 1)
        y = tmp["Tref"].to_numpy()

        model = LinearRegression().fit(X, y)
        y_pred = (model.coef_[0] * X.ravel()) + model.intercept_

        # Datos
        fig.add_trace(
            go.Scatter(
                x=X.ravel(),
                y=y,
                mode="markers",
                marker=dict(color=_SCATTER_BASE_COLOR),
                name="Datos",
                visible=(i == 0),
            )
        )
        # Ajuste
        fig.add_trace(
            go.Scatter(
                x=X.ravel(),
                y=y_pred,
                mode="lines",
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
                name="Ajuste lineal",
                visible=(i == 0),
            )
        )

    buttons = []
    for i, col in enumerate(columns):
        vis = [False] * (2 * len(columns))
        vis[2 * i] = True
        vis[2 * i + 1] = True
        buttons.append({"label": col, "method": "update", "args": [{"visible": vis}]})

    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        xaxis_title="Temperatura de termopar (°C)",
        yaxis_title="Temperatura de referencia (°C)",
    )
    return fig
