import os
import sys

sys.path.append('/Users/shahrozkhan/Documents/Compute Maritime/code/DesignApp')
from DesignApp.physical_paras import get_physical_paras, get_heighAtAP_shaftheight

import numpy as np
from scipy.spatial import ConvexHull
from pathlib import Path
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State



def generate_design(file_path: str) -> np.ndarray:
    """
    Generate a design based on the given file path.

    Args:
    - file_path (str): Path to the file containing design points.

    Returns:
    - np.array: 3D design array.
    """
    design_points = np.loadtxt(file_path, delimiter=',')
    patch_1 = np.transpose(design_points[:19 * 42]).reshape(3, 42, 19)[:, :, :-1]
    patch_5 = patch_1[:, -1, :].reshape(3, 1, 18)
    design_points = design_points[19 * 42:]

    patch_2 = np.transpose(design_points[:53 * 42]).reshape(3, 42, 53)
    patch_5 = np.concatenate((patch_5, patch_2[:, -1, :-1].reshape(3, 1, 52)), axis=2)
    design_points = design_points[53 * 42:]

    patch_3 = np.transpose(design_points[:29 * 22]).reshape(3, 22, 29)
    patch_5 = np.concatenate((patch_5, patch_3[:, -1, :-1].reshape(3, 1, 28)), axis=2)
    design_points = design_points[29 * 22:]

    patch_4 = np.transpose(design_points[:3 * 21]).reshape(3, 21, 3)
    patch_4 = np.concatenate((np.zeros([3, 1, 3]), patch_4), axis=1)
    patch_4[:, :, 0] = patch_3[:, :, -1]
    patch_3 = patch_3[:, :, 1:-1]
    patch_4[1, :, -1] = 0
    patch_4[0, 0, :] = patch_4[0, 0, 0]
    patch_5 = np.concatenate((patch_5, patch_4[:, -1, :].reshape(3, 1, 3)), axis=2)
    design_points = design_points[3 * 21:]

    patch_5 = np.concatenate((patch_5, np.transpose(design_points).reshape(3, -1, 101)), axis=1)

    patch_2, patch_3 = equalize_matrices(patch_2, patch_3)
    patch_2, patch_4 = equalize_matrices(patch_2, patch_4)
    design_3d = np.concatenate((patch_1, patch_2, patch_3, patch_4), axis=2)
    design_3d = np.concatenate((design_3d, patch_5), axis=1)

    return design_3d


def equalize_matrices(mat1: np.ndarray, mat2: np.ndarray) -> tuple:
    """
    Given two 3D matrices, equalize their size along axis 1 by repeating the last row
    of the smaller matrix.

    Parameters:
    - mat1, mat2: The 3D matrices.

    Returns:
    - mat1_new, mat2_new: The resized matrices.
    """

    size1 = mat1.shape[1]
    size2 = mat2.shape[1]

    # If mat1 is smaller, increase its size
    if size1 < size2:
        difference = size2 - size1
        last_row = mat1[:, 0, :]  # Extract the last row
        repeated_rows = np.repeat(last_row[:, np.newaxis, :], difference, axis=1)
        mat1_new = np.concatenate([repeated_rows, mat1], axis=1)
        mat2_new = mat2

    # If mat2 is smaller, increase its size
    elif size2 < size1:
        difference = size1 - size2
        last_row = mat2[:, 0, :]  # Extract the last row
        repeated_rows = np.repeat(last_row[:, np.newaxis, :], difference, axis=1)
        mat2_new = np.concatenate([repeated_rows, mat2], axis=1)
        mat1_new = mat1

    # If both are of the same size
    else:
        mat1_new = mat1
        mat2_new = mat2

    return mat1_new, mat2_new


def create_shaft(shaft_height, shaft_length, shaft_shape_file):
    """
    Revolve a curve defined by the x and z coordinates from a file around the x-axis to create a 3D shaft shape.
    The function also extends the shaft by a defined length and calculates its approximate volume.

    Parameters:
    - shaft_height (float): The height to which the shaft curve should be translated before revolving.

    Returns:
    - numpy.ndarray: A 3D array with meshgrid arrays of x, y, and z coordinates of the revolved and extended shaft.
    - float: The approximate volume of the created shaft.

    Note:
    The function loads the x and z coordinates of the curve from a file with a fixed path.
    """
    shaft_shape = np.loadtxt(shaft_shape_file, delimiter=',')

    x = shaft_shape[:, 0] + 3.720
    z = shaft_shape[:, 2] + shaft_height

    # Translate the curve so the starting point is on the x-axis

    z_translated = z - z[0]

    v = np.linspace(0, 2 * np.pi, 500)

    X, V = np.meshgrid(x, v)
    Z_original, _ = np.meshgrid(z_translated, v)

    Y = Z_original * np.sin(V)
    Z = Z_original * np.cos(V) + z[0]

    X = np.concatenate((X, shaft_length + X[:, -1].reshape(-1, 1)), axis=1)
    Y = np.concatenate((Y, Y[:, -1].reshape(-1, 1)), axis=1)
    Z = np.concatenate((Z, Z[:, -1].reshape(-1, 1)), axis=1)

    # Compute the approximate volume using the shell method
    delta_x = np.diff(X[0])  # Change in x between two consecutive points

    max_y = np.max(np.abs(Y), axis=0)

    # Compute the volume for each disc and sum them up
    disc_volumes = np.pi * max_y[:-1] ** 2 * delta_x
    volume = np.sum(disc_volumes)

    return np.array([X, Y, Z]), volume


def get_physical_parameters_data(physical_parameters, attribute_id, value_id):
    if physical_parameters is None:
        return [
            {attribute_id: "Volume", value_id: "-"},
            {attribute_id: "LCB", value_id: "-"},
            {attribute_id: "Beam", value_id: "-"},
            {attribute_id: "Transom Height", value_id: "-"},
            {attribute_id: "Height at AP", value_id: "-"},
            {attribute_id: "Shaft Height", value_id: "-"},
            {attribute_id: "Engine Point x-axis", value_id: "-"}
        ]
    else:
        return [
            {attribute_id: "Volume", value_id: round(physical_parameters[0], 3)},
            {attribute_id: "LCB", value_id: round(physical_parameters[1], 3)},
            {attribute_id: "Beam", value_id: round(physical_parameters[2], 3)},
            {attribute_id: "Transom Height", value_id: round(physical_parameters[3], 3)},
            {attribute_id: "Height at AP", value_id: round(physical_parameters[4], 3)},
            {attribute_id: "Shaft Height", value_id: round(physical_parameters[5], 3)},
            {attribute_id: "Engine Point x-axis", value_id: round(physical_parameters[6], 3)}
        ]


def convert_array_to_csv(data: np.ndarray) -> str:
    """
    Convert a numpy array of physical parameters into a CSV string format.

    Parameters:
    - data (np.ndarray): A 1D numpy array containing the physical parameters.
                          Expected order: [Volume, LCB, Beam, Transom Height, Height at AP, Shaft Height, Engine Point x-axis]

    Returns:
    - str: A formatted CSV string ready for exporting or saving.

    Example:
    Input: np.array([124513.76, 122.59, 20.08, 13.54, 11.80, 4.16, 15.11])
    Output: 'Volume,LCB,Beam,Transom Height,Height at AP,Shaft Height,Engine Point x-axis\n124513.760,122.590,20.080,13.540,11.800,4.160,15.110\n'
    """
    headers = ["Volume", "LCB", "Beam", "Transom Height", "Height at AP", "Shaft Height", "Engine Point x-axis"]
    formatted_data = ", ".join(["{:.3f}".format(val) for val in data])
    csv_string = ", ".join(headers) + "\n" + formatted_data + "\n"
    return csv_string


# Initial indices
i, j = 0, 0


def build_app(file_directory: Path) -> dash.Dash:
    """Build the Dash app."""
    physical_parameters = get_physical_parameters_data(None, "Attribute", "Value")

    t_sen = np.loadtxt(file_directory / "t-sen.txt", delimiter=",")
    trainData = t_sen[:1372, :]
    genData = t_sen[1372:, :]
    scatter_1 = go.Scatter(
        x=trainData[:, 0],
        y=trainData[:, 1],
        mode='markers',
        name='training Designs',
        marker=dict(color='black')
    )
    scatter_2 = go.Scatter(
        x=genData[:, 0],
        y=genData[:, 1],
        mode='markers',
        name='New Designs',
        marker=dict(color='red')
    )
    ch_trainData = ConvexHull(trainData)
    ch_genData = ConvexHull(genData)
    sca_ch_trainData = go.Scatter(
        x=trainData[ch_trainData.vertices, 0],
        y=trainData[ch_trainData.vertices, 1],
        mode='lines',
        line=dict(color='black'),
        name='training Designs'
    )
    sca_ch_genData = go.Scatter(
        x=genData[ch_genData.vertices, 0],
        y=genData[ch_genData.vertices, 1],
        mode='lines',
        line=dict(color='red'),
        name='New Designs'
    )

    def plot_design(deisgn_3D: np.ndarray, shaft_3D: np.ndarray, fig_title: str) -> tuple:
        """Create a 3D plot based on the given design."""
        num_contours = 10
        range_x = deisgn_3D[0, :, :].max() - deisgn_3D[0, :, :].min()
        size_x = range_x / (2 * num_contours)
        range_y = deisgn_3D[1, :, :].max() - deisgn_3D[1, :, :].min()
        size_y = range_y / num_contours
        range_z = deisgn_3D[2, :, :].max() - deisgn_3D[2, :, :].min()
        size_z = range_z / num_contours

        gray_scale = [[0, '#808080'], [1, '#808080']]
        hull_1 = go.Surface(
            x=deisgn_3D[0, :, :],
            y=deisgn_3D[1, :, :],
            z=deisgn_3D[2, :, :],
            colorscale=gray_scale,
            contours={
                "x": {"show": True, "start": deisgn_3D[0, :, :].min(), "end": deisgn_3D[0, :, :].max(),
                      "size": size_x},
                "y": {"show": True, "start": deisgn_3D[1, :, :].min(), "end": deisgn_3D[1, :, :].max(),
                      "size": size_y},
                "z": {"show": True, "start": deisgn_3D[2, :, :].min(), "end": deisgn_3D[2, :, :].max(),
                      "size": size_z}
            },
            showscale=False
        )
        hull_2 = go.Surface(
            x=deisgn_3D[0, :, :],
            y=-deisgn_3D[1, :, :],
            z=deisgn_3D[2, :, :],
            colorscale=gray_scale,
            showscale=False
        )
        shaft_srf = go.Surface(x=shaft_3D[0], y=shaft_3D[1], z=shaft_3D[2], colorscale=gray_scale, showscale=False)

        data = [hull_1, hull_2, shaft_srf, scatter_1, scatter_2, sca_ch_trainData,
                sca_ch_genData]  # Add the scatter plot data to the list of traces

        layout = go.Layout(
            title=fig_title,
            title_font=dict(size=24, family="Arial, bold"),  # Set the title to be bold and adjust the size as required
            title_x=0.5,  # This will center the title
            title_y=0.95,  # Adjust the title's vertical position if necessary
            scene=dict(
                aspectmode='data',
                xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',
                           backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',
                           backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0)"),
                zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',
                           backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0)"),
                bgcolor="rgba(0,0,0,0)",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0.5, y=0, z=0),
                    eye=dict(x=2, y=1.8, z=1.8)
                )
            ),
            xaxis=dict(
                domain=[0, 0.3],
                anchor='y',
            ),
            yaxis=dict(
                domain=[0.7, 1],
                anchor='x',
                title='t-SEN Plot'
            ),
        )
        return data, layout

    app = dash.Dash(__name__)

    # Define the app layout
    app.layout = html.Div([
        dcc.Loading(
            id="loading",
            type="circle",  # can also use "default" or "cube"
            children=[
                html.Div([
                    dcc.Graph(id='3d-plot-top', style={'width': '100%', 'height': '90vh'}),
                    html.Div(id='design-number-top', children=f"Design Number (Without geometric Operators): {i}/1000"),
                    # Display the design number
                    html.Button('Previous Design', id='prev-button-top', style={
                        'width': '200px',
                        'height': '50px',
                        'fontSize': '20px',
                        'marginRight': '10px'
                    }),
                    html.Button('Next Design', id='next-button-top', style={
                        'width': '200px',
                        'height': '50px',
                        'fontSize': '20px',
                        'marginRight': '30px'
                    }),
                    html.Button('Physical Parameters', id='eval-button-top', style={
                        'width': '200px',
                        'height': '50px',
                        'fontSize': '20px',
                        'marginRight': '10px'
                    }),
                    html.Button('Export Parameters', id='export-button-top', style={
                        'width': '200px',
                        'height': '50px',
                        'fontSize': '20px',
                    }),
                    dcc.Download(id="download-csv-top"),
                    dash.dash_table.DataTable(
                        id='title-table',
                        columns=[
                            {"name": "Physical Parameters", "id": "Attribute"},
                            {"name": "Value", "id": "Value"}
                        ],
                        data=physical_parameters,
                        style_table={'width': '50%'},
                        style_cell={'textAlign': 'center'},
                        style_header={
                            'backgroundColor': 'lightgrey',
                            'fontWeight': 'bold'
                        }
                    )
                ])
            ]),
        # Spacer Div to create empty space
        html.Div(style={'height': '60px'}),
        dcc.Loading(
            id="loading-bottom",
            type="circle",  # can also use "default" or "cube"
            children=[
                html.Div([
                    dcc.Graph(id='3d-plot-bottom', style={'width': '100%', 'height': '90vh'}),
                    html.Div(id='design-number-bottom', children=f"Design Number (With geometric Operators): {i}/1000"),
                    # Display the design number
                    html.Button('Previous Design', id='prev-button-bottom', style={
                        'width': '200px',
                        'height': '50px',
                        'fontSize': '20px',
                        'marginRight': '10px'
                    }),
                    html.Button('Next Design', id='next-button-bottom', style={
                        'width': '200px',
                        'height': '50px',
                        'fontSize': '20px',
                        'marginRight': '30px'
                    }),
                    html.Button('Physical Parameters', id='eval-button-bottom', style={
                        'width': '200px',
                        'height': '50px',
                        'fontSize': '20px',
                        'marginRight': '10px'
                    }),
                    html.Button('Export Parameters', id='export-button-bottom', style={
                        'width': '200px',
                        'height': '50px',
                        'fontSize': '20px',
                    }),
                    dcc.Download(id="download-csv-bottom"),
                    dash.dash_table.DataTable(
                        id='title-table-below',
                        columns=[
                            {"name": "Physical Parameters", "id": "Attribute-below"},
                            {"name": "Value", "id": "Value-below"}
                        ],
                        data=physical_parameters,
                        style_table={'width': '50%'},
                        style_cell={'textAlign': 'center'},
                        style_header={
                            'backgroundColor': 'lightgrey',
                            'fontWeight': 'bold'
                        }
                    )
                ])
            ])
    ])

    @app.callback(
        [Output('3d-plot-top', 'figure'), Output('design-number-top', 'children'), Output('title-table', 'data')],
        [Input('next-button-top', 'n_clicks'), Input('prev-button-top', 'n_clicks'),
         Input('eval-button-top', 'n_clicks')],
        [State('3d-plot-top', 'relayoutData')]
    )
    def update_figure_top(next_clicks_top, prev_clicks_top, eval_button_top, relayout_data_top):
        """Update the top figure."""
        global i
        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'next-button-top':
            i += 1
        elif button_id == 'prev-button-top' and i > 0:
            i -= 1

        file_path = file_directory / f"Design Points_{i}.txt"
        design = generate_design(file_path)

        # Shaft creation and volume calculation
        _, shaft_height = get_heighAtAP_shaftheight(design)
        shaft, shaft_volume = create_shaft(shaft_height, 5, file_directory / "shaft cap.csv")

        if button_id == 'eval-button-top':
            para_values = get_physical_paras(design, 15)
            para_values[0] = para_values[0] + shaft_volume  # adding the shaft volume to the submerged volume
            physical_parameters = get_physical_parameters_data(para_values, "Attribute", "Value")
        else:
            physical_parameters = get_physical_parameters_data(None, "Attribute", "Value")

        plot_title = "Designs from 15-dimensional latent space (Without geometric operators - 55% reduction)"
        data, layout = plot_design(design, shaft, plot_title)

        # Check if there's previous view state
        if relayout_data_top and 'scene.camera' in relayout_data_top:
            layout['scene']['camera'] = relayout_data_top['scene.camera']

        fig = go.Figure(data=data, layout=layout)
        return fig, f"Design Number (Without geometric Operators): {i + 1}/1000", physical_parameters

    # Callback to handle the button click and return CSV data for download
    @app.callback(
        Output('download-csv-top', 'data'),
        [Input('export-button-top', 'n_clicks')],
    )
    def export_paras_top(n_clicks):
        if n_clicks is None:
            # Prevents the callback from being triggered on page load
            raise dash.exceptions.PreventUpdate
        file_path = file_directory / f"Design Points_{i}.txt"
        design = generate_design(file_path)
        para_values = get_physical_paras(design, 15)

        _, shaft_volume = create_shaft(para_values[5], 5, file_directory / "shaft cap.csv")
        para_values[0] = para_values[0] + shaft_volume

        csv_string = convert_array_to_csv(para_values)
        return dict(type="text/csv", content=csv_string, filename=f"physical_parameters_design_{i + 1}.csv")

    @app.callback(
        [Output('3d-plot-bottom', 'figure'), Output('design-number-bottom', 'children'),
         Output('title-table-below', 'data')],
        [Input('next-button-bottom', 'n_clicks'), Input('prev-button-bottom', 'n_clicks'),
         Input('eval-button-bottom', 'n_clicks')],
        [State('3d-plot-bottom', 'relayoutData')]
    )
    def update_figure_bottom(next_clicks_bottom, prev_clicks_bottom, eval_button_bottom, relayout_data_bottom):
        """Update the bottom figure."""
        global j
        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'next-button-bottom':
            j += 1
        elif button_id == 'prev-button-bottom' and j > 0:
            j -= 1

        file_path = file_directory / f"Design Points_{j + 1001}.txt"
        design = generate_design(file_path)

        # Shaft creation and volume calculation
        _, shaft_height = get_heighAtAP_shaftheight(design)
        shaft, shaft_volume = create_shaft(shaft_height, 5, file_directory / "shaft cap.csv")

        if button_id == 'eval-button-bottom':
            para_values = get_physical_paras(design, 15)
            para_values[0] = para_values[0] + shaft_volume  # adding the shaft volume to the submerged volume
            physical_parameters = get_physical_parameters_data(para_values, "Attribute-below", "Value-below")
        else:
            physical_parameters = get_physical_parameters_data(None, "Attribute-below", "Value-below")

        plot_title = "Designs from 12-dimensional latent space (With Geometric Operators - 64% reduction)"
        data, layout = plot_design(design, shaft, plot_title)

        # Check if there's previous view state
        if relayout_data_bottom and 'scene.camera' in relayout_data_bottom:
            layout['scene']['camera'] = relayout_data_bottom['scene.camera']

        fig = go.Figure(data=data, layout=layout)
        return fig, f"Design Number (With geometric Operators): {j + 1}/1000", physical_parameters

    # Callback to handle the button click and return CSV data for download
    @app.callback(
        Output('download-csv-bottom', 'data'),
        [Input('export-button-bottom', 'n_clicks')],
    )
    def export_paras_bottom(n_clicks):
        if n_clicks is None:
            # Prevents the callback from being triggered on page load
            raise dash.exceptions.PreventUpdate
        file_path = file_directory / f"Design Points_{j + 1001}.txt"
        design = generate_design(file_path)
        para_values = get_physical_paras(design, 15)

        _, shaft_volume = create_shaft(para_values[5], 5, file_directory / "shaft cap.csv")
        para_values[0] = para_values[0] + shaft_volume

        csv_string = convert_array_to_csv(para_values)
        return dict(type="text/csv", content=csv_string, filename=f"physical_parameters_design_{j + 1}.csv")

    return app

#if __name__ == '__main__':
#    app = build_app()
#    app.run_server(debug=True)
#    app.run_server(debug=True, host="0.0.0.0")
