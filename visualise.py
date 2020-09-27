
import plotly.graph_objects as go
import numpy as np

from functions import *

def plot_kappa(grid, domain, nu, returnfigs=False, show=True):
    """"""""" Plots real and imaginary parts of kappa """
    # Grid params
    x, y, delx = grid['x'], grid['y'], grid['delx']  # grid dimensions in metres
    M, N, L = grid['M'], grid['N'], grid['L']  # Grid dimensions in cell numbers
    kappa = ConvertToKappa(domain, nu)

    fig1 = go.Figure(data=go.Heatmap(z=np.real(kappa).reshape(M, N), y=np.linspace(0, N, N), x=np.linspace(0, M, M),
                                     colorscale="magma", showscale=True))
    fig1.update_layout(
        legend=dict(traceorder='reversed', font_size=16), showlegend=True, width=800, height=800,
        autosize=False, title=go.layout.Title(text="Real part of kappa"), titlefont=dict(size=30))

    xaxis = go.layout.XAxis(
        title=go.layout.xaxis.Title(text="x Distance (m)", font=dict(size=18))),

    yaxis = go.layout.YAxis(
        title=go.layout.yaxis.Title(text="y Distance (m)", font=dict(size=18)))
    if show:
        fig1.show()

    heatmap = go.Heatmap(z=np.imag(kappa).reshape(M, N), y=np.linspace(0, N, N), x=np.linspace(0, M, M),
                         colorscale="magma", showscale=True)

    xaxis = go.layout.XAxis(
        title=go.layout.xaxis.Title(text="x Distance (m)", font=dict(size=18))),

    yaxis = go.layout.YAxis(
        title=go.layout.yaxis.Title(text="y Distance (m)", font=dict(size=18)))
    title = "Imaginary part of kappa"
    layout = go.Layout(
        legend=dict(traceorder='reversed', font_size=16), showlegend=True, width=800, height=800,
        autosize=False, title=go.layout.Title(text=title), titlefont=dict(size=30))
    fig = go.Figure(data=[heatmap], layout=layout)
    if show:
        fig.show()
    if returnfigs:
        return fig1, fig


def plot_E(grid, E, inc_SourceMeasure=False, Normalise=False):
    """"""""" Plots real and imaginary parts of E-field """
    # Grid params
    x, y, delx = grid['x'], grid['y'], grid['delx']  # grid dimensions in metres
    M, N, L, P = grid['M'], grid['N'], grid['L'], grid['P']  # Grid dimensions in cell numbers

    E_shaped = E.reshape(M, N)
    E_max = np.max(np.abs(E_shaped[P:-P, P:-P]))
    E_shaped_norm = E_shaped / E_max
    if not Normalise:
        E_shaped_norm = E_shaped.copy()

    fig = go.Figure(
        data=go.Heatmap(z=np.real(E_shaped_norm), y=np.linspace(0, N, N), x=np.linspace(0, M, M), colorscale='Balance',
                        showscale=True))
    fig.update_layout(
        legend=dict(traceorder='reversed', font_size=16), showlegend=True,
        width=700 * (E_shaped.shape[1] / E_shaped.shape[0]), height=700,  # Scale width so ratio matches height
        autosize=False, title=go.layout.Title(text="Finite Differences - Real"), titlefont=dict(size=30))

    xaxis = go.layout.XAxis(
        title=go.layout.xaxis.Title(text="x Distance (m)", font=dict(size=18))),

    yaxis = go.layout.YAxis(
        title=go.layout.yaxis.Title(text="y Distance (m)", font=dict(size=18)))
    fig.show()

    fig = go.Figure(
        data=go.Heatmap(z=np.abs(E_shaped_norm), y=np.linspace(0, N, N), x=np.linspace(0, M, M), colorscale="Balance",
                        zmin=0.0,
                        zmax=1.0, showscale=True))

    # if inc_SourceMeasure:
    #     # Plot measurement position
    #     measurevec = np.zeros(L)
    #     measurevec[N_measure], measurevec[N_source] = 1, 2
    #     measurement = go.Heatmap(z=np.reshape(measurevec, [M, N]), showscale=False, opacity=0.7)
    #     fig.add_trace(measurement)

    fig.update_layout(
        legend=dict(traceorder='reversed', font_size=16), showlegend=True,
        width=700 * (E_shaped.shape[1] / E_shaped.shape[0]), height=700,  # Scale width so ratio matches height
        autosize=False, title=go.layout.Title(text="Finite Differences - Absolute Value"), titlefont=dict(size=30))

    xaxis = go.layout.XAxis(
        title=go.layout.xaxis.Title(text="x Distance (m)", font=dict(size=18))),

    yaxis = go.layout.YAxis(
        title=go.layout.yaxis.Title(text="y Distance (m)", font=dict(size=18)))
    fig.show()