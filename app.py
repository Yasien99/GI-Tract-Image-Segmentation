import streamlit as st
from streamlit.logger import get_logger
from Utils.Data_processing import *


plt.rcdefaults()
plt.style.use("seaborn-darkgrid")
color_list = mpl.rcParams["axes.prop_cycle"]
color_list = list(color_list.by_key().values())[0]


LOGGER = get_logger(__name__)



def run():
    st.set_page_config(
        page_title="GI-TRACT-IMAGE-Segmentation",
        page_icon="💻",
        layout="centered",
    )
    # st.write('## this is our Work Documentation ✈️', anchor='#header1')
    # st.header("## Table of Contents")


if __name__ == "__main__":
    run()
    st.write("# GI-TRACT-IMAGE-Segmentation")
    
    with st.sidebar:
        st.write("# Model Selection")
        st.write("---" * 50) 
        
        #st.markdown("""---""")
    read_data = st.sidebar.checkbox("read_data", False)
    
    if read_data:
        df = pd.read_csv(r"Data\train.csv")
        df = rename_df(df)
        df_train, train_ids, valid_ids, test_ids = read_data_path(df)
        (
            train_generator,
            val_generator,
            train_generator0,
            val_generator0,
            train_generator1,
            val_generator1,
            train_generator2,
            val_generator2,
        ) = data_gene(df_train, train_ids, valid_ids, test_ids)
        
        
        
    plt.style.use('ggplot')

#     st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 200px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 500px;
#         margin-left: -200px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

    Model_hist, Inspection, Real_masks,= st.tabs(["Model History","Inspection", "Real Masks" ])
    #Inspection.subheader("real masks")

    with Model_hist.container():
        #st.write("## Model History")
        history_1 = pd.read_csv('csv/history_effientNet.csv')
        # history_2 = pd.read_csv('csv/history_omar.csv')
        history_2 = pd.read_csv('csv/history1.csv')
        # history_2.rename(columns={'jacard_coef': 'dice_coef', 'val_jacard_coef': 'val_dice_coef'}, inplace=True)
        
        col1, col2 = st.columns(2)
        with col1:
            hist = st.selectbox("Model History", ['effientNet', 'U-Net'])
        with col2:
            if hist == 'effientNet':
                mode = st.selectbox("Plot Mode", list(history_1.columns[1:8]))
            elif hist == 'U-Net':
            #     mode = st.selectbox("Plot Mode", list(history_2.columns[1:8]))
            # elif hist == 'MOdel2':
                mode = st.selectbox("Plot Mode", list(history_2.columns[1:4]))
                
        
        
        show_df = st.sidebar.checkbox("Show Dataframe", False)
        if show_df:
            st.dataframe(history_1)
            # st.markdown("""---""")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            grid = st.checkbox("Grid", True)
        with col2:
            animation = st.checkbox("Animation", True)
            
        themes =['ggplot2', 'seaborn', 'simple_white', 'plotly',
         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         'ygridoff', 'gridon']
        
        
        st.sidebar.write("---" * 20) 
        # with st.container():
            # st.sidebar.write("## Theme")
        theme = st.sidebar.selectbox('Theme', themes, index = 5, )
        
        if hist == 'effientNet':
            animate_plot(history_1, theme, Plot_mode=mode , y_range_loss_range=1, delay=100, Animate=animation, SHOW_GRID=grid)
        elif hist  == 'U-Net':
            animate_plot(history_2, theme, Plot_mode=mode , y_range_loss_range=1, delay=100, Animate=animation, SHOW_GRID=grid, title = mode + ' after 25 epochs')
        # elif hist == 'MOdel2':
        #     animate_plot(history_3, theme, Plot_mode=mode , y_range_loss_range=1, delay=100, Animate=animation, SHOW_GRID=grid, title = mode + ' after 25 epochs')
        
        if read_data:
            with Inspection.container(): 
                st.write("## Real masks")
                fig = Inspect_data(df_train)
                st.pyplot(fig)
    
    #Visuals.subheader("predicted masks")
            with Real_masks.container():
                st.write("## Predicted masks")
                model = tf.keras.models.load_model('Models\model0_nocompile.h5', compile = False)
                model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef,f1_score,f2_score,precision,recall,iou_score])
                fig = Plot_predicte_masks(model,df_train,valid_ids)
                st.pyplot(fig)
        

        
        #model = tf.keras.models.load_model('Models\model0_nocompile.h5', compile = False)
        #model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef,f1_score,f2_score,precision,recall,iou_score])
        #Plot_history(model,df_train,valid_ids)
    