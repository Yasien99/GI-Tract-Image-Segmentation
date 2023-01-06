import plotly.graph_objects as go
import pandas as pd
import streamlit as st


def animate_plot(
    hisotry,
    theme = 'plotly_dark',
    Plot_mode="dice_coef",
    y_range_loss_range=1,
    delay=50,
    Animate=True,
    streamlit=True,
    SHOW_GRID=True,
    title=None,
    autorange = False,
):
    trace1_col = Plot_mode
    trace2_col = "val_" + Plot_mode
    hisotry = pd.DataFrame((hisotry.copy()))
    if hisotry.columns[0] != "epoch":
        hisotry[hisotry.columns[0]] = hisotry[hisotry.columns[0]].apply(lambda x: x + 1)
        hisotry.rename(columns={hisotry.columns[0]: "epoch"}, inplace=True)
    # print(hisotry.columns)

    if y_range_loss_range == None:
        y_range_max_range = max(hisotry[trace1_col].max(), hisotry[trace2_col].max())
    else:
        y_range_max_range = y_range_loss_range
    hisotry[trace1_col] = hisotry[trace1_col].apply(lambda x: round(x, 4))
    hisotry[trace2_col] = hisotry[trace2_col].apply(lambda x: round(x, 4))

    trace1 = go.Scatter(
        x=hisotry[hisotry.columns[0]], y=hisotry[trace1_col], name="train_" + trace1_col
    )
    trace2 = go.Scatter(
        x=hisotry[hisotry.columns[0]], y=hisotry[trace2_col], name=trace2_col
    )
    if Animate:
        fig = go.Figure(
            data=[trace1, trace2],
            layout=go.Layout(
                # template='plotly_dark',
                xaxis=dict(
                    range=[1, hisotry[hisotry.columns[0]].max()],
                    autorange=False,
                ),
                yaxis=dict(range=[0, y_range_max_range], autorange=False),
                # * buttons
                showlegend=True,
                hovermode="x unified",
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": delay, "redraw": False},
                                        "fromcurrent": True,
                                        "transition": {
                                            "duration": 50,
                                            "easing": "quadratic-in-out",
                                        },
                                    },
                                ],
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ],
                    )
                ],
            ),
            # * frames
            frames=[
                dict(
                    data=[
                        dict(
                            type="scatter",
                            x=hisotry["epoch"][: k + 1],
                            y=hisotry[trace1_col][: k + 1],
                        ),
                        dict(
                            type="scatter",
                            x=hisotry["epoch"][: k + 1],
                            y=hisotry[trace2_col][: k + 1],
                        ),
                    ],
                    traces=[0, 1],
                )
                for k in range(2, len(hisotry["epoch"]) + 1)
            ],
        )

    else:
        fig = go.Figure(
            data=[trace1, trace2],
            layout=go.Layout(
                xaxis=dict(
                    range=[1, hisotry[hisotry.columns[0]].max()],
                    autorange=False,
                ),
                yaxis=dict(range=[0, y_range_max_range], autorange=False),
                # * buttons
                showlegend=True,
                hovermode="x unified",
            ),
        )

    # fig.update_layout(xaxis=dict(rangeselector = dict(font = dict( color = "black"))))
    # update button color

    # fig.layout.update(template='plotly_dark')
    # fig.update_layout(
    #     xaxis_title="Epochs",
    #     yaxis_title=Plot_mode,
    #     title_x=0.5,
    # )
    # * change xtick
    if title == None:
        fig.update_layout(
            title=Plot_mode + " after Transfer learning",
            xaxis_title="Epochs",
            yaxis_title=Plot_mode,
            title_x=0.5,
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title="Epochs",
            yaxis_title=Plot_mode,
            title_x=0.5,
        )
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1))
    fig.update_layout(yaxis=dict(tickmode="linear", tick0=0, dtick=0.1))
    if autorange == True:
        fig.update_layout(yaxis= dict(autorange = True))
    fig.update_layout(xaxis=dict(showgrid=SHOW_GRID), yaxis=dict(showgrid=SHOW_GRID))
    fig.update_layout(template=theme)

    if streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()
