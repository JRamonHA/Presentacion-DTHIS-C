import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression


def scatter_plot(filepath, columns, traces_colors=None, x_label=None, y_label=None, legend_labels=None):
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
    pio.renderers.default = "iframe"

    # Si no se proporcionan etiquetas de la leyenda, se usan los nombres de las columnas
    if legend_labels is None or len(legend_labels) != len(columns):
        legend_labels = columns

    # Lista de colores por defecto si no se proporciona una lista de colores
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
              "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    if traces_colors is None or len(traces_colors) != len(columns):
        traces_colors = colors[:len(columns)]

    fig = go.Figure()

    # Agregar cada traza con su respectivo color y etiqueta
    for i, col in enumerate(columns):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='markers',
            marker=dict(color=traces_colors[i]),
            name=legend_labels[i]
        ))

    fig.update_layout(
        xaxis_title=x_label if x_label else 'Índice',
        yaxis_title=y_label if y_label else ', '.join(columns),
        hovermode='x unified'
    )

    return fig


def scatter_subplots(filepath, subplot_configs):
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
    # Cargar datos
    data = pd.read_parquet(filepath)
    pio.renderers.default = "iframe"
    
    num_subplots = len(subplot_configs)
    # Crear subplots en una sola columna, compartiendo eje x
    fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True)
    
    # Colores por defecto
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
                      "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
    
    for i, config in enumerate(subplot_configs, start=1):
        cols = config.get('columns', [])
        traces_colors = config.get('traces_colors', colors[:len(cols)])
        x_label = config.get('x_label', 'Índice')
        y_label = config.get('y_label', ', '.join(cols))
        legend_labels = config.get('legend_labels', cols)
        
        for j, col in enumerate(cols):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode='markers',
                marker=dict(color=traces_colors[j]),
                name=legend_labels[j]
            ), row=i, col=1)
        
        # Actualiza la etiqueta del eje Y para este subplot
        fig.update_yaxes(title_text=y_label, row=i, col=1)
    
    # Actualiza la etiqueta del eje X solo en el último subplot (compartido)
    fig.update_xaxes(title_text=x_label, row=num_subplots, col=1)
    fig.update_layout(hovermode='x unified')
    
    return fig


def columns_plot(filepath, x_col, y_col, x_label, y_label, trace_color=None, legend_name=None, identity_line=False, x_range=None, y_range=None):
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
    pio.renderers.default = "iframe"

    fig = go.Figure()
    display_name = legend_name if legend_name is not None else None

    # Scatter principal
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='markers',
        marker=dict(color=trace_color or '#636EFA'),
        name=display_name
    ))

    # Línea identidad y ejes iguales si se solicita
    if identity_line:
        lo = min(data[x_col].min(), data[y_col].min())
        hi = max(data[x_col].max(), data[y_col].max())
        # Ajustar según rangos manuales, si existen
        if x_range:
            lo = min(lo, x_range[0])
            hi = max(hi, x_range[1])
        if y_range:
            lo = min(lo, y_range[0])
            hi = max(hi, y_range[1])

        fig.add_trace(go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode='lines',
            line=dict(dash='dash'),
            name='Pendiente (m=1)'
        ))
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[lo, hi])

    # Aplicar rangos manuales si se proporcionan
    if x_range:
        fig.update_xaxes(range=list(x_range))
    if y_range:
        fig.update_yaxes(range=list(y_range))

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=True
    )

    return fig


def columns_subplots(filepath, subplot_configs, identity_line=False):
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
    pio.renderers.default = "iframe"

    num_subplots = len(subplot_configs)
    fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=False)

    for i, config in enumerate(subplot_configs, start=1):
        x = data[config['x_col']]
        y = data[config['y_col']]
        x_label = config.get('x_label', config['x_col'])
        y_label = config.get('y_label', config['y_col'])
        trace_color = config.get('trace_color', '#636EFA')
        legend_name = config.get('legend_name', config['y_col'])

        # Agregar datos
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(color=trace_color),
                name=legend_name
            ),
            row=i, col=1
        )

        # Etiquetas
        fig.update_xaxes(title_text=x_label, row=i, col=1)
        fig.update_yaxes(title_text=y_label, row=i, col=1)

        # Rango de ejes: manual o calculado
        x_range = config.get('x_range', None)
        y_range = config.get('y_range', None)

        if x_range:
            fig.update_xaxes(range=x_range, row=i, col=1)
        if y_range:
            fig.update_yaxes(range=y_range, row=i, col=1)

        # Línea identidad
        if identity_line:
            # Determinar rango para la línea identidad
            x_lo = x_range[0] if x_range else x.min()
            x_hi = x_range[1] if x_range else x.max()
            y_lo = y_range[0] if y_range else y.min()
            y_hi = y_range[1] if y_range else y.max()
            line_lo = min(x_lo, y_lo)
            line_hi = max(x_hi, y_hi)

            # Color alternado
            identity_colors = ['#EF553B', '#FFA15A']
            identity_color = identity_colors[(i - 1) % len(identity_colors)]

            fig.add_trace(
                go.Scatter(
                    x=[line_lo, line_hi],
                    y=[line_lo, line_hi],
                    mode='lines',
                    line=dict(dash='dash', color=identity_color),
                    name='Pendiente (m=1)'
                ),
                row=i, col=1
            )

    fig.update_layout(
        showlegend=True,
        height=400 * num_subplots
    )
    return fig


def dropdowns_plot(filepath, trace_labels, x_cols, y_cols, x_axis_title, y_axis_title):
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
    
    pio.renderers.default = "iframe"

    colors = ["#AB63FA", "#19D3F3", "#FFA15A", "#636EFA", "#00CC96",
              "#EF553B", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    fig = go.Figure()
    
    n_traces = len(trace_labels)
    # Agregar cada traza con su respectivo color (se reinicia la cuenta si hay más de 10 trazas)
    for i in range(n_traces):
        fig.add_trace(go.Scatter(
            x=data[x_cols[i]],
            y=data[y_cols[i]],
            mode='markers',
            marker=dict(color=colors[i % len(colors)]),
            name=trace_labels[i]
        ))
    
    for i in range(n_traces):
        fig.data[i].visible = (i == 0)
    
    # Crear botones para el menú desplegable
    buttons = []
    for i in range(n_traces):
        visibility = [True if j == i else False for j in range(n_traces)]
        button = {
            "label": trace_labels[i],
            "method": "update",
            "args": [{"visible": visibility}]
        }
        buttons.append(button)
    
    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top"
            }
        ],
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title
    )
    
    return fig


def compare_plot(filepath, group_dict, x_label="Tiempo", y_label=None, legend_labels=None):
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

    pio.renderers.default = "iframe"

    # Definir los pares de colores:
    group_colors = [
        ("#636EFA", "#AB63FA"),  
        ("#00CC96", "#19D3F3"),
        ("#EF553B", "#FFA15A"),   
        ("#FF6692", "#B6E880"),  
        ("#FF97FF", "#FECB52")  
    ]
    num_color_groups = len(group_colors)

    fig = go.Figure()
    group_labels = list(group_dict.keys())
    traces_per_group = 2
    total_traces = len(group_labels) * traces_per_group

    # Agregar trazas para cada grupo
    for i, group in enumerate(group_labels):
        sensor_col, ref_col = group_dict[group]
        sensor_color, ref_color = group_colors[i % num_color_groups]

        # Obtener nombres personalizados para la leyenda si se proporcionan
        sensor_legend = legend_labels[group][0] if legend_labels and group in legend_labels else sensor_col
        ref_legend = legend_labels[group][1] if legend_labels and group in legend_labels else ref_col

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[sensor_col],
            mode='markers',
            marker=dict(color=sensor_color),
            name=sensor_legend
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[ref_col],
            mode='markers',
            marker=dict(color=ref_color),
            name=ref_legend
        ))

    # Inicialmente, solo se muestran las trazas del primer grupo
    for j in range(total_traces):
        fig.data[j].visible = (j < traces_per_group)

    # Si no se proporciona `y_label`, se usan los nombres de los menús
    if y_label is None:
        y_label = {group: group for group in group_labels}

    # Crear botones para el menú desplegable
    buttons = []
    for i, group in enumerate(group_labels):
        visibility = [False] * total_traces
        start = i * traces_per_group
        visibility[start] = True
        visibility[start + 1] = True

        button = {
            "label": group,
            "method": "update",
            "args": [
                {"visible": visibility},
                {"yaxis": {"title": {"text": y_label.get(group, group)}}}  # Cambia el título dinámicamente
            ]
        }
        buttons.append(button)

    fig.update_layout(
        hovermode='x unified',
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.0,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top"
        }],
        xaxis_title=x_label,
        yaxis_title=y_label.get(group_labels[0], group_labels[0])  # Título inicial del eje Y
    )

    return fig


def linear_reg_subplots(filepath, columns, time_intervals, titles):
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
    # Configurar renderizador
    pio.renderers.default = "iframe"
    
    # Validar longitud de títulos
    if len(columns) != len(titles):
        raise ValueError("La cantidad de títulos debe ser igual a la cantidad de columnas.")

    # Cargar datos y verificar el índice
    data = pd.read_parquet(filepath)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError('El índice del DataFrame debe ser de tipo DatetimeIndex.')

    # Crear columna Tref y asignar valores según intervalos de tiempo
    data['Tref'] = np.nan
    for start, end, tref_value in time_intervals:
        indices = data.between_time(start, end).index
        data.loc[indices, 'Tref'] = tref_value

    # Filtrar datos con Tref definido
    data_all = data.dropna(subset=['Tref'])
    
    # Determinar distribución de subplots
    n = len(columns)
    n_rows = math.ceil(math.sqrt(n))
    n_cols = math.ceil(n / n_rows)
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols, shared_xaxes=True, shared_yaxes=True,
        subplot_titles=titles  # Asignar los títulos personalizados
    )

    # Paleta de colores para las líneas de regresión
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
              "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    # Iterar sobre cada columna para ajustar el modelo y agregar trazas
    for idx, col in enumerate(columns):
        temp = data_all[[col, 'Tref']].dropna()
        if temp.empty:
            continue
        X = temp[col].values.reshape(-1, 1)
        y = temp['Tref'].values
        
        model = LinearRegression()
        model.fit(X, y)
        coef = model.coef_[0]
        intercept = model.intercept_
        y_pred = coef * X.flatten() + intercept

        # Determinar posición del subplot
        row = idx // n_cols + 1
        col_subplot = idx % n_cols + 1

        # Agregar gráfico de dispersión con leyenda en tooltip
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y,
            mode='markers',
            marker=dict(color='#0D2A63'),
            name=f'Datos {col}',  # Se muestra solo en tooltip
            hoverinfo='x+y+name'  # Mostrar información en tooltip
        ), row=row, col=col_subplot)

        # Agregar línea de regresión con color asignado y leyenda en tooltip
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y_pred,
            mode='lines',
            line=dict(color=colors[idx % len(colors)], width=2),
            name=f'Ajuste {col}',  # Se muestra solo en tooltip
            hoverinfo='x+y+name'
        ), row=row, col=col_subplot)

    # Actualizar ejes de todos los subplots
    fig.update_xaxes(tickmode='linear', dtick=10)
    fig.update_yaxes(tickmode='linear', dtick=10)

    # Asignar títulos a ejes en la parte inferior y en la izquierda
    for j in range(1, n_cols+1):
        fig.update_xaxes(title_text="Temperatura de termopar (°C)", row=n_rows, col=j)
    for i in range(1, n_rows+1):
        fig.update_yaxes(title_text="T<sub>ref</sub> (°C)", row=i, col=1)

    # Ocultar cuadro de leyendas, pero mantenerlas en el tooltip
    fig.update_layout(showlegend=False)
    
    return fig


def linear_reg_plot(filepath, columns, time_intervals):
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

    pio.renderers.default = "iframe"

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError('El índice del DataFrame debe ser de tipo DatetimeIndex.')

    # Crear columna Tref con valores nulos
    data['Tref'] = np.nan
    for start, end, tref_value in time_intervals:
        indices = data.between_time(start, end).index
        data.loc[indices, 'Tref'] = tref_value

    # Filtrar datos con Tref
    data_all = data.dropna(subset=['Tref'])

    # Paleta de colores Plotly
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
              "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    fig = go.Figure()

    # Calcular regresiones y agregar trazas
    for i, col in enumerate(columns):
        temp = data_all[[col, 'Tref']].dropna()
        X = temp[col].values.reshape(-1, 1)
        y = temp['Tref'].values
        model = LinearRegression()
        model.fit(X, y)
        coef = model.coef_[0]
        intercepto = model.intercept_
        y_pred = coef * X.flatten() + intercepto

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y,
            mode='markers',
            marker=dict(color='#0D2A63'),
            name='Datos',
            visible=(i == 0)
        ))

        # Línea de regresión (color cíclico)
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y_pred,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            name='Ajuste lineal',
            visible=(i == 0)
        ))

    # Crear menú desplegable
    buttons = []
    for i, col in enumerate(columns):
        visibility = [False] * (2 * len(columns))
        visibility[2*i] = True  # Activar scatter
        visibility[2*i + 1] = True  # Activar línea de regresión

        buttons.append({
            "label": col,
            "method": "update",
            "args": [{"visible": visibility}]
        })

    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.0,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top"
        }],
        xaxis_title="Temperatura de termopar (°C)",
        yaxis_title="Temperatura de referencia (°C)"
    )

    return fig


def calculate_error(filepath, col1, col2, resample_interval=None):
    """
    Calcula el Error Medio (ME) y el Error Absoluto Medio (MAE) entre dos columnas.
    Opcionalmente, permite resamplear el DataFrame antes de calcular los errores.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - col1 (str): Nombre de la primera columna.
    - col2 (str): Nombre de la segunda columna.
    - resample_interval (str, opcional): Intervalo de resampleo (ej. '10s'). Si no se proporciona, no se resamplea.

    Retorna:
    - None: Imprime el ME y el MAE
    """
    data = pd.read_parquet(filepath)

    if col1 not in data.columns or col2 not in data.columns:
        raise ValueError(f'Las columnas {col1} y {col2} no existen en el DataFrame.')

    if resample_interval:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError('El índice del DataFrame debe ser de tipo DatetimeIndex para resamplear.')
        data = data.resample(resample_interval).mean()

    error = data[col1] - data[col2]
    ME = round(error.mean(), 4)
    MAE = round(error.abs().mean(), 4)

    print(f'ME: {ME}')
    print(f'MAE: {MAE}')


def wind_calibration(filepath, ws='WS', v='V', t='T', resample_interval=None):
    """
    Ajusta la ecuación de calibración del Wind Sensor.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - ws (str): Nombre de la columna con la velocidad del viento (WS).
    - v (str): Nombre de la columna con el voltaje ajustado (V).
    - t (str): Nombre de la columna con la temperatura (T).
    - resample_interval (str, opcional): Intervalo de resampleo (ej. '10s'). Si no se proporciona, no se resamplea.

    Retorna:
    - None: Imprime parámetros de calibración con a, b, c y r2, y la ecuación de calibración.
    """
    data = pd.read_parquet(filepath)

    if not all(col in data.columns for col in [ws, v, t]):
        raise ValueError('Una o más columnas especificadas no existen en el DataFrame.')

    # Ajustar el voltaje restándole 1.1621
    data[v] = data[v] - 1.1621

    if resample_interval:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError('El índice del DataFrame debe ser de tipo DatetimeIndex para resamplear.')
        data = data.resample(resample_interval).mean()

    data_filtered = data[(data[ws] > 0) & (data[v] > 0) & (data[t] > 0)]
    X = np.log(data_filtered[[v, t]])
    y = np.log(data_filtered[ws])
    modelo = LinearRegression()
    modelo.fit(X, y)
    intercepto = modelo.intercept_
    coeficientes = modelo.coef_
    a = np.exp(intercepto)
    b = coeficientes[0]
    c = coeficientes[1]
    equation = f'WS = {a:.4f} * {v}^({b:.4f}) * {t}^({c:.4f})'
    r2 = modelo.score(X, y)

    r2_r = round(r2, 4)

    print(f"Correlación: {r2_r}")
    print(f"{equation}")


def thermo_calibration(filepath, columns, time_intervals):
    """
    Realiza ecuaciones de calibración para termopares a partir 
    de columnas específicas de un DataFrame e intervalos de tiempo, 
    asignando valores a 'Tref' y realizando regresiones.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - columns (list): Lista de nombres de columnas para la regresión.
    - time_intervals (list): Lista de tuplas (inicio, fin, valor_Tref).

    Retorna:
    - None: Imprime las ecuaciones de calibración para cada columna.
    """
    data = pd.read_parquet(filepath)

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError('El índice del DataFrame debe ser de tipo DatetimeIndex.')

    for col in columns:
        if col not in data.columns:
            raise ValueError(f'La columna \'{col}\' no existe en el DataFrame.')

    data['Tref'] = np.nan
    for start, end, tref_value in time_intervals:
        indices = data.between_time(start, end).index
        data.loc[indices, 'Tref'] = tref_value

    data_all = data.dropna(subset=['Tref'])
    for col in columns:
        temp = data_all[[col, 'Tref']].dropna()
        X = temp[col].values.reshape(-1, 1)
        y = temp['Tref'].values
        modelo = LinearRegression()
        modelo.fit(X, y)
        coef = modelo.coef_[0]
        intercepto = modelo.intercept_

        print(f'{col}:')
        print(f'Tref = {coef:.4f} * {col} + {intercepto:.4f}\n')