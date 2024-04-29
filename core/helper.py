from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot(
    x: list[list | np.ndarray | pd.Series],
    y: list[list | np.ndarray | pd.Series],
    series_names: list[str],
    x2: list[list | np.ndarray | pd.Series] = None,
    y2: list[list | np.ndarray | pd.Series] = None,
    series_names2: list[str] = None,
    x_label: str = None,
    y_label: str = None,
    y2_label: str = None,
    title: str = None,
    hover_data: dict[str, list] = None,
    x_range: tuple = None,
    y_range: tuple = None,
    y2_range: tuple = None,
    x_error: list[list | np.ndarray | pd.Series] = None,
    y_error: list[list | np.ndarray | pd.Series] = None,
    y2_error: list[list | np.ndarray | pd.Series] = None,
    lines: bool = False,
    markers: bool = True,
    show_legend: bool = True,
    template: str = "plotly",
    show_fig: bool = True,
    return_fig: bool = False,
    x_log: bool = False,
    y_log: bool = False,
    y2_log: bool = False,
    line_style: list[str] = None,
    line_color: list[str] = None,
    **kwargs,
) -> None | go.Figure:
    """
    Displays/returns a scatter plot generated using plotly for any arbitrary set of data. 
    Supports multiple traces/series on the primary y-axis or on a secondary y-axis.

    Parameters
    ----------
    x : list[list  |  np.ndarray  |  pd.Series]
        X-axis data corresponding to primary y-axis data
    y : list[list  |  np.ndarray  |  pd.Series]
        Primary y-axis data
    series_names : list[str]
        Names of traces/series for primary y-axis data
    x2 : list[list  |  np.ndarray  |  pd.Series], optional
        X-axis data corresponding to secondary y-axis data, by default None
    y2 : list[list  |  np.ndarray  |  pd.Series], optional
        Secondary y-axis data, by default None
    series_names2 : list[str], optional
        Names of traces/series for secondary y-axis data, by default None
    x_label : str, optional
        X-axis label, by default None
    y_label : str, optional
        Primary y-axis label, by default None
    y2_label : str, optional
        Secondary y-axis label, by default None
    title : str, optional
        Plot title, by default None
    hover_data : dict[str, list], optional
        Data to be displayed when data point is hovered, by default None
    x_range : tuple, optional
        X-axis plot data range, by default None
    y_range : tuple, optional
        Primary y-axis plot data range, by default None
    y2_range : tuple, optional
        Seconday y-axis plot data range, by default None
    x_error : list[list  |  np.ndarray  |  pd.Series], optional
        X-axis error data by default None
    y_error : list[list  |  np.ndarray  |  pd.Series], optional
        Primary y-axis error data, by default None
    y2_error : list[list  |  np.ndarray  |  pd.Series], optional
        Secondary y-axis error data, by default None
    lines : bool, optional
        Boolean on whether the plot displays lines, by default False
    markers : bool, optional
        Boolean on whether the plot displays markers, by default True
    show_legend : bool, optional
        Boolean on whether the legend is shown, by default True
    template : str, optional
        Plotly template, by default 'plotly'
    show_fig : bool, optional
        Boolean on whether the fig object will be displayed, by default True
    return_fig : bool, optional
        Boolean on whether the fig object will be returned, by default False
    x_log : bool, optional
        Boolean on whether x-axis will be plotted in log scale, by default False
    y_log : bool, optional
        Boolean on whether primariy y-axis will be plotted in log scale, by default False
    y2_log : bool, optional
        Boolean on whether secondary y-axis will be plotted in log scale, by default False
    line_style : list[str], optional
        specifies the style of the line. Can be dash, solid or other dash options found at https://plotly.com/python-api-reference/generated/plotly.graph_objects.scatter.html#plotly.graph_objects.scatter.Line, by default None
    line_color : list[str], optional
        color used to represent each line with str being a hex string, a named CSS color or other color options found at https://plotly.com/python-api-reference/generated/plotly.graph_objects.scatter.html#plotly.graph_objects.scatter.Line, by default None

    Returns
    -------
    None | go.Figure
        Plotly graph_objects figure if return_fig is set to True. Else returns None.

    Examples
    --------
    Creating a plot of a vertical and horizontal line.

    >>> fig = plot(x=[[0,1,2], [0,0,0]], y=[[0,0,0], [0,1,2]], series_names=['Horizontal Line', 'Vertical Line'], \
    ... lines=True, markers=False, show_fig=False, return_fig=True)
    >>> print(fig)
    Figure({
        'data': [{'connectgaps': True,
                'hovertemplate': '(%{x}, %{y})<br><b>%{meta[0]}</b><extra></extra>',
                'marker': {'size': 8},
                'meta': [Horizontal Line],
                'mode': 'lines',
                'name': 'Horizontal Line',
                'type': 'scatter',
                'x': [0, 1, 2],
                'xaxis': 'x',
                'y': [0, 0, 0],
                'yaxis': 'y'},
                {'connectgaps': True,
                'hovertemplate': '(%{x}, %{y})<br><b>%{meta[0]}</b><extra></extra>',
                'marker': {'size': 8},
                'meta': [Vertical Line],
                'mode': 'lines',
                'name': 'Vertical Line',
                'type': 'scatter',
                'x': [0, 0, 0],
                'xaxis': 'x',
                'y': [0, 1, 2],
                'yaxis': 'y'}],
        'layout': {'showlegend': True,
                'template': '...',
                'title': {},
                'xaxis': {'anchor': 'y', 'domain': [0.0, 0.94]},
                'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0]},
                'yaxis2': {'anchor': 'x', 'overlaying': 'y', 'side': 'right'}}
    })

    """

    def validate_type_list(**kwargs) -> None:
        """Validate that all args are of type list."""

        for input_name, item in kwargs.items():
            if type(item) != list:
                raise Exception(f'Input "{input_name}" needs to be of type list.')

        return

    def validate_equal_len(**kwargs) -> None:
        """Validate that all args have equal len."""

        input_names = list(kwargs.keys())
        is_correct_len = len(set([len(data) for data in kwargs.values()])) == 1
        if not is_correct_len:
            raise Exception(
                f'Inputs "{", ".join(input_names)}" needs to be the same length.'
            )

        return

    def process_plot_data(x: list, y: list, series_names: list) -> tuple:
        """Prepares plot data for figure plotting."""

        validate_type_list(x=x, y=y, series_names=series_names)
        validate_equal_len(x=x, y=y, series_names=series_names)
        x = [list(x_item) for x_item in x]
        y = [list(y_item) for y_item in y]

        return x, y, series_names

    def get_mode(lines: bool, markers: bool) -> str:
        """Returns lines and/or markers mode for figure plotting."""

        modes = []
        if lines:
            modes.append("lines")
        if markers:
            modes.append("markers")
        if len(modes) == 0:
            raise Exception(
                'Please set at least one mode to True: "lines" and/or "markers".'
            )

        return "+".join(modes)

    def get_meta(series_names: list, index: int, hover_data: dict, y: list) -> list:
        """Returns list of meta data for which each element is displayed when the corresponding data point is hovered upon."""

        if hover_data is None:
            return [series_names[index]]

        series_names_hover = [series_names[index] for _ in y[index]]
        zip_func = partial(zip, series_names_hover)

        for _, data in hover_data.items():
            zip_func = partial(zip_func, data[index])

        return list(zip_func())

    def get_hover_template(hover_data: dict) -> str:
        """Returns the hover template format for figure plotting."""

        hover_template = "(%{x}, %{y})<br><b>%{meta[0]}</b>"
        if hover_data is not None:
            for i, label in enumerate(hover_data.keys(), 1):
                hover_template = (
                    hover_template + f"<br><b>{label}: </b>" + "%{meta" + f"[{i}]" + "}"
                )

        return hover_template + "<extra></extra>"

    def create_fig() -> go.Figure:
        """Creates plotly figure."""

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for index, _ in enumerate(y):
            fig.add_trace(
                go.Scatter(
                    x=x[index],
                    y=y[index],
                    name=series_names[index],
                    error_x=x_error[index] if x_error else None,
                    error_y=y_error[index] if y_error else None,
                    meta=get_meta(series_names, index, hover_data, y),
                    hovertemplate=get_hover_template(hover_data),
                    mode=get_mode(lines, markers),
                    marker_size=8,
                    connectgaps=True,
                    line=dict(dash=line_style[index]) if line_style != None else None,
                    line_color=line_color[index] if line_color != None else None,
                ),
                secondary_y=True if y2 and y[index] in y2 else False,
            )

        fig.update_layout(title=title, showlegend=show_legend)
        fig.update_layout(template=template)
        fig.update_xaxes(
            title_text=x_label, range=x_range, type="log" if x_log else None
        )
        fig.update_yaxes(
            title_text=y_label,
            secondary_y=False,
            range=y_range,
            type="log" if y_log else None,
        )
        fig.update_yaxes(
            title_text=y2_label,
            secondary_y=True,
            range=y2_range,
            type="log" if y2_log else None,
        )

        return fig

    x, y, series_names = process_plot_data(x, y, series_names)

    if y2:
        x2, y2, series_names2 = process_plot_data(x2, y2, series_names2)
        x.extend(x2)
        y.extend(y2)
        series_names.extend(series_names2)
        if y_error and y2_error:
            y_error.extend(y2_error)

    fig = create_fig()

    if show_fig:
        fig.show()

    if return_fig:
        return fig

    return
