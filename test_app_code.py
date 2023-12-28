import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline

def interpolate_3d_points(x, y, z, num_points=100):
    """
    Interpolates a 3D curve using cubic spline interpolation.
    """
    
    # Create parameter t (assuming equal spacing for simplicity)
    t = np.linspace(0, 1, len(x))

    # Create cubic spline for each dimension
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    cs_z = CubicSpline(t, z)

    # Evaluate spline over a finer grid
    t_fine = np.linspace(0, 1, num_points)
    x_fine = cs_x(t_fine)
    y_fine = cs_y(t_fine)
    z_fine = cs_z(t_fine)

    return x_fine, y_fine, z_fine

def generate_design(deisgn_3D, length = 250.082, beam = 42.836/2, draft=23.416):
    deisgn_3D[0, :, :] = gaussian_filter(deisgn_3D[0, :, :], sigma=1)
    deisgn_3D[1, :, :] = gaussian_filter(deisgn_3D[1, :, :], sigma=1)
    deisgn_3D[2, :, :] = gaussian_filter(deisgn_3D[2, :, :], sigma=1)
    deisgn_3D[1, 0, :] = 0
    deisgn_3D[1, :, 0] = 0

    deisgn_3D[0, :, :] = deisgn_3D[0, :, :] + np.max(np.absolute(deisgn_3D[0, :, :]))
    deisgn_3D[2, :, :] = deisgn_3D[2, :, :] + np.max(np.absolute(deisgn_3D[2, :, :]))

    deisgn_3D[0, :, :] = deisgn_3D[0, :, :] * (length/np.max(np.absolute(deisgn_3D[0, :, :])))
    deisgn_3D[1, :, :] = deisgn_3D[1, :, :] * (beam/np.max(np.absolute(deisgn_3D[1, :, :])))
    deisgn_3D[2, :, :] = deisgn_3D[2, :, :] * (draft/np.max(np.absolute(deisgn_3D[2, :, :])))

    # Fixing keel line
    idx = 30 + np.where(np.diff(deisgn_3D[2, 0, 29:]) < 0)[0]  
    while idx.size > 0:
        deisgn_3D[2, 0, idx[0]] = deisgn_3D[2, 0, idx[0] - 1]
        idx = 30 + np.where(np.diff(deisgn_3D[2, 0, 29:]) < 0)[0]   
    
    # fixing each section at the flat of bottom
    for i in range(deisgn_3D.shape[2]):
        idx = np.where(np.diff(deisgn_3D[2, :, i]) < 0)[0]
        while idx.size > 0:
            deisgn_3D[2, idx[0]+1, i] = deisgn_3D[2, idx[0], i]
            idx = np.where(np.diff(deisgn_3D[2, :, i]) < 0)[0]

    # fixing sections close to the stern - flat of side
    idx = 40 + np.where(np.diff(deisgn_3D[1, -1, 39:]) > 0)[0]
    while idx.size > 0:
        deisgn_3D[1, :, idx[0]] = deisgn_3D[1, :, idx[0]-1]
        idx = 40 + np.where(np.diff(deisgn_3D[1, -1, 39:]) > 0)[0]

    #fixing the bow
    idx = np.where(np.diff(deisgn_3D[0, -1, :10]) > 0)[0]
    for index in idx:
        deisgn_3D[0, :, index] = deisgn_3D[0, :, idx[-1]+1]

    design_3D_ = np.zeros((deisgn_3D.shape[0], 1000, deisgn_3D.shape[2]))
    for i in range(deisgn_3D.shape[2]):
        design_3D_[0, :, i], design_3D_[1, :, i], design_3D_[2, :, i] = interpolate_3d_points(deisgn_3D[0, :, i], deisgn_3D[1, :, i], deisgn_3D[2, :, i], num_points=1000)

    return design_3D_

def plot_design(deisgn_3D: np.ndarray, fig_title: str):
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
        showscale=False,
        lighting=dict(ambient=0.4, diffuse=0.6, fresnel=2, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=100, z=1000)
    )
    hull_2 = go.Surface(
        x=deisgn_3D[0, :, :],
        y=-deisgn_3D[1, :, :],
        z=deisgn_3D[2, :, :],
        colorscale=gray_scale,
        showscale=False,
        lighting=dict(ambient=0.4, diffuse=0.6, fresnel=2, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=100, z=1000)
    )

    data = [hull_1, hull_2]  # Add the scatter plot data to the list of traces

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
        )
    )
    return data, layout

app = dash.Dash(__name__)
server = app.server
# Initial variables
np.random.seed(42)

i = 0

latentSpace = loadmat('latentSpace.mat')
KLM = latentSpace['KLM']
KLS = latentSpace['KLS']
KLU = latentSpace['KLU']
KLV = latentSpace['KLV']

initial_logs = "Message logs:\n"

# Update the layout with the chat/text box, send button, and message logs
app.layout = html.Div([
    dcc.Loading(
        id="loading",
        type="cube",  # can also use "default" or "cube"
        children=[
            html.Div([
                html.Div([
                    # This div will contain the 3D plot
                    html.Div([
                        dcc.Graph(id='3d-plot', style={'height': '90vh'}),
                    ], style={'display': 'inline-block', 'width': '85%'}),
        
                    # This div will contain the sliders
                    html.Div([
                        html.Div([
                            html.Label('Parameter 1'),
                            dcc.Slider(id='slider-1', min=KLS[0,0], max=KLS[0,1], step=0.01, value=KLS[0,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[0,0], KLS[0,1], 3)}),
                        ], style={'padding': '5px'}),
        
                        html.Div([
                            html.Label('Parameter 2'),
                            dcc.Slider(id='slider-2', min=KLS[1,0], max=KLS[1,1], step=0.01, value=KLS[1,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[1,0], KLS[1,1], 3)}),
                        ], style={'padding': '5px'}),
        
                        html.Div([
                            html.Label('Parameter 3'),
                            dcc.Slider(id='slider-3', min=KLS[2,0], max=KLS[2,1], step=0.01, value=KLS[2,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[2,0], KLS[2,1], 3)}),
                        ], style={'padding': '5px'}),
        
                        html.Div([
                            html.Label('Parameter 4'),
                            dcc.Slider(id='slider-4', min=KLS[3,0], max=KLS[3,1], step=0.01, value=KLS[3,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[3,0], KLS[3,1], 3)}),
                        ], style={'padding': '5px'}),
        
                        html.Div([
                            html.Label('Parameter 5'),
                            dcc.Slider(id='slider-5', min=KLS[4,0], max=KLS[4,1], step=0.01, value=KLS[4,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[4,0], KLS[4,1], 3)}),
                        ], style={'padding': '5px'}),
                        
                        html.Div([
                            html.Label('Parameter 6'),
                            dcc.Slider(id='slider-6', min=KLS[5,0], max=KLS[5,1], step=0.01, value=KLS[5,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[5,0], KLS[5,1], 3)}),
                        ], style={'padding': '5px'}),
        
                        html.Div([
                            html.Label('Parameter 7'),
                            dcc.Slider(id='slider-7', min=KLS[6,0], max=KLS[6,1], step=0.01, value=KLS[6,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[6,0], KLS[6,1], 3)}),
                        ], style={'padding': '5px'}),
        
                        html.Div([
                            html.Label('Parameter 8'),
                            dcc.Slider(id='slider-8', min=KLS[7,0], max=KLS[7,1], step=0.01, value=KLS[7,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[7,0], KLS[7,1], 3)}),
                        ], style={'padding': '5px'}),
        
                        html.Div([
                            html.Label('Parameter 9'),
                            dcc.Slider(id='slider-9', min=KLS[8,0], max=KLS[8,1], step=0.01, value=KLS[8,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[8,0], KLS[8,1], 3)}),
                        ], style={'padding': '5px'}),
        
                        html.Div([
                            html.Label('Parameter 10'),
                            dcc.Slider(id='slider-10', min=KLS[9,0], max=KLS[9,1], step=0.01, value=KLS[9,0], marks={i: {'label': str(i), 'style': {'color': 'rgba(0, 0, 0, 0)'}} for i in np.linspace(KLS[9,0], KLS[9,1], 3)}),
                        ], style={'padding': '5px'}),
                    ], style={'display': 'inline-block', 'width': '15%', 'vertical-align': 'top'}),
                ], style={'display': 'flex', 'justifyContent': 'center'}),
        
                html.Div([
                    html.Div([
                        html.Label('Length:', style={'margin-right': '10px'}),
                        dcc.Input(id='length-input', type='number', placeholder='Length', value=250.082, style={'margin-right': '20px'}),
        
                        html.Label('Beam:', style={'margin-right': '10px'}),
                        dcc.Input(id='beam-input', type='number', placeholder='Beam', value=42.836 / 2, style={'margin-right': '20px'}),
        
                        html.Label('Draft:', style={'margin-right': '10px'}),
                        dcc.Input(id='draft-input', type='number', placeholder='Draft', value=23.416, style={'margin-right': '20px'}),
        
                        html.Button('Random Design', id='generate-button', style={'height': '30px', 'fontSize': '16px'}),
                    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
                ]),
                html.Div(id='design-number', children="Design Number: 1"),
                html.Div([
                    dcc.Input(
                        id='message-input',
                        type='text',
                        placeholder='Message ShipHullGAN...',
                        style={'height': '25px', 'width': '60%', 'margin-top': '10px', 'margin-right': '10px', 'fontSize': '15px'}
                    ),
                    html.Button('Send Message', id='send-button', style={'height': '30px', 'fontSize': '16px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                html.Div(
                    dcc.Textarea(
                        id='message-logs',
                        value="Message logs:\n",
                        style={'width': '70%', 'height': '100px', 'margin-top': '10px', 'fontSize': '16px', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'},
                        readOnly=True
                    ),
                    style={'textAlign': 'center'}
                ),
            ])
        ]
    )
])


@app.callback(
    [Output('3d-plot', 'figure'), 
     Output('design-number', 'children'),
     Output('slider-1', 'value'),
     Output('slider-2', 'value'),
     Output('slider-3', 'value'),
     Output('slider-4', 'value'),
     Output('slider-5', 'value'),
     Output('slider-6', 'value'),
     Output('slider-7', 'value'),
     Output('slider-8', 'value'),
     Output('slider-9', 'value'),
     Output('slider-10', 'value')],
    [Input('generate-button', 'n_clicks'),
     Input('slider-1', 'value'),
     Input('slider-2', 'value'),
     Input('slider-3', 'value'),
     Input('slider-4', 'value'),
     Input('slider-5', 'value'),
     Input('slider-6', 'value'),
     Input('slider-7', 'value'),
     Input('slider-8', 'value'),
     Input('slider-9', 'value'),
     Input('slider-10', 'value')],
    [State('3d-plot', 'relayoutData'), 
     State('length-input', 'value'),
     State('beam-input', 'value'), 
     State('draft-input', 'value')]
)
def update_figure(generate_clicks, slider1_value, slider2_value, slider3_value, slider4_value, slider5_value, slider6_value,  slider7_value, slider8_value, slider9_value, slider10_value, relayout_data, length, beam, draft):
    global i
    global design
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'generate-button':
            i += 1
            # Generate new design when button is clicked
            design = np.random.uniform(KLS[:,0], KLS[:,1], (1, KLS.shape[0])).flatten()
            slider1_value = np.clip(design[0], KLS[0,0], KLS[0,1]) 
            slider2_value = np.clip(design[1], KLS[1,0], KLS[1,1]) 
            slider3_value = np.clip(design[2], KLS[2,0], KLS[2,1])
            slider4_value = np.clip(design[3], KLS[3,0], KLS[3,1])
            slider5_value = np.clip(design[4], KLS[4,0], KLS[4,1])
            slider6_value = np.clip(design[5], KLS[5,0], KLS[5,1])
            slider7_value = np.clip(design[6], KLS[6,0], KLS[6,1])
            slider8_value = np.clip(design[7], KLS[7,0], KLS[7,1])
            slider9_value = np.clip(design[8], KLS[8,0], KLS[8,1])
            slider10_value = np.clip(design[9], KLS[9,0], KLS[9,1])
        else:
            # Use current slider values to update the design
            design[:10] = np.array([slider1_value, slider2_value, slider3_value, slider4_value, slider5_value, 
            slider6_value, slider7_value, slider8_value, slider9_value, slider10_value])
    else:
        # Default design if no trigger (e.g., initial load)
        design = np.random.uniform(KLS[:,0], KLS[:,1], (1, KLS.shape[0])).flatten()
        slider1_value = np.clip(design[0], KLS[0,0], KLS[0,1]) 
        slider2_value = np.clip(design[1], KLS[1,0], KLS[1,1]) 
        slider3_value = np.clip(design[2], KLS[2,0], KLS[2,1])
        slider4_value = np.clip(design[3], KLS[3,0], KLS[3,1])
        slider5_value = np.clip(design[4], KLS[4,0], KLS[4,1])
        slider6_value = np.clip(design[5], KLS[5,0], KLS[5,1])
        slider7_value = np.clip(design[6], KLS[6,0], KLS[6,1])
        slider8_value = np.clip(design[7], KLS[7,0], KLS[7,1])
        slider9_value = np.clip(design[8], KLS[8,0], KLS[8,1])
        slider10_value = np.clip(design[9], KLS[9,0], KLS[9,1])

    if length is None:
        length = 250.082
    if beam is None:
        beam = 42.836 / 2
    if draft is None:
        draft = 23.416

    design_3D = (np.dot(design, np.transpose(KLM)) + KLU) * KLV
    design_3D = generate_design(np.transpose(design_3D.reshape(3,54,21), (0, 2, 1)), length, beam, draft)

    plot_title = "ShipHullGAN"

    data, layout = plot_design(design_3D, plot_title)

    # Check if there's previous view state
    if relayout_data and 'scene.camera' in relayout_data:
        layout['scene']['camera'] = relayout_data['scene.camera']

    fig = go.Figure(data=data, layout=layout)
    #fig.add_trace(sca_ch_genData)
    return fig, f"Design Number: {i + 1}", slider1_value, slider2_value, slider3_value, slider4_value, slider5_value, slider6_value, slider7_value, slider8_value, slider9_value, slider10_value

# Callback for updating message logs
@app.callback(
    Output('message-logs', 'value'),
    [Input('send-button', 'n_clicks')],
    [State('message-input', 'value'),
     State('message-logs', 'value')]
)
def update_logs(n_clicks, input_value, logs_value):
    global i
    if n_clicks and input_value:
        # Append the new message to the existing log
        new_log = logs_value + f"\n{i} - {input_value}"
        return new_log
    else:
        # Return the existing log if no new message is entered
        return logs_value

if __name__ == '__main__':
    app.run_server(debug=True)
