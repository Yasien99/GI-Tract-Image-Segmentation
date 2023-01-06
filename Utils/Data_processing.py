from Utils.Utils import *
from Utils.Model_utils import *
from Utils.Modification import *
import streamlit as st

# df = pd.read_csv(r'Data\train.csv')


def rename_df(df):
    df.rename(columns={"class": "class_name"}, inplace=True)
    # --------------------------------------------------------------------------
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
    return df


def read_data_path(df):
    TRAIN_DIR = "Data/train"
    # get all the image path with glob
    all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
    x = all_train_images[0].rsplit("/", 4)[
        0
    ]  ## ../input/uw-madison-gi-tract-image-segmentation/train

    path_partial_list = []
    for i in range(0, df.shape[0]):
        path_partial_list.append(
            os.path.join(
                x,
                "case" + str(df["case"].values[i]),
                "case"
                + str(df["case"].values[i])
                + "_"
                + "day"
                + str(df["day"].values[i]),
                "scans",
                "slice_" + str(df["slice"].values[i]),
            )
        )
    df["path_partial"] = path_partial_list
    df["path_partial"] = df["path_partial"].apply(
        lambda x: x.replace("Data", "Data/train")
    )
    # * --------------------------------------------------------------------------
    # *--------------------------------------------------------------------------
    path_partial_list = []
    for i in range(0, len(all_train_images)):
        path_partial_list.append(str(all_train_images[i].rsplit("_", 4)[0]))

    tmp_df = pd.DataFrame()
    tmp_df["path_partial"] = path_partial_list
    tmp_df["path"] = all_train_images
    # * --------------------------------------------------------------------------
    # *--------------------------------------------------------------------------
    df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
    df["width"] = df["path"].apply(lambda x: int(x[:-3].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-3].rsplit("_", 4)[2]))
    del x, path_partial_list, tmp_df

    # * --------------------------------------------------------------------------
    # *--------------------------------------------------------------------------
    # RESTRUCTURE  DATAFRAME
    df_train = pd.DataFrame({"id": df["id"][::3]})

    df_train["large_bowel"] = df["segmentation"][::3].values
    df_train["small_bowel"] = df["segmentation"][1::3].values
    df_train["stomach"] = df["segmentation"][2::3].values

    df_train["path"] = df["path"][::3].values
    df_train["case"] = df["case"][::3].values
    df_train["day"] = df["day"][::3].values
    df_train["slice"] = df["slice"][::3].values
    df_train["width"] = df["width"][::3].values
    df_train["height"] = df["height"][::3].values

    df_train.reset_index(inplace=True, drop=True)
    df_train.fillna("", inplace=True)
    # the count columns store the number of mask the exist in image
    df_train["count"] = np.sum(df_train.iloc[:, 1:4] != "", axis=1).values

    # * --------------------------------------------------------------------------
    # *--------------------------------------------------------------------------

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(
        skf.split(X=df_train, y=df_train["count"], groups=df_train["case"]), 1
    ):
        df_train.loc[val_idx, "fold"] = fold

    df_train["fold"] = df_train["fold"].astype(np.uint8)

    test_ids = df_train[df_train["count"] == 0].index
    train_ids = df_train[
        (df_train["fold"] != fold_selected) & (~df_train.index.isin(test_ids))
    ].index
    valid_ids = df_train[
        (df_train["fold"] == fold_selected) & (~df_train.index.isin(test_ids))
    ].index

    df_train.groupby("fold").size()

    return df_train, train_ids, valid_ids, test_ids


def data_gene(df_train, train_ids, valid_ids, test_ids):
    train_generator = DataGenerator(
        df_train[df_train.index.isin(train_ids)], shuffle=True
    )
    val_generator = DataGenerator(df_train[df_train.index.isin(valid_ids)])
    # test_generator = DataGenerator(df_train[df_train.index.isin(test_ids)])
    # every mask seperated
    train_generator0 = DataGenerator1D(
        0, df_train[df_train.index.isin(train_ids)], shuffle=True
    )
    val_generator0 = DataGenerator1D(0, df_train[df_train.index.isin(valid_ids)])

    train_generator1 = DataGenerator1D(
        1, df_train[df_train.index.isin(train_ids)], shuffle=True
    )
    val_generator1 = DataGenerator1D(1, df_train[df_train.index.isin(valid_ids)])

    train_generator2 = DataGenerator1D(
        2, df_train[df_train.index.isin(train_ids)], shuffle=True
    )
    val_generator2 = DataGenerator1D(2, df_train[df_train.index.isin(valid_ids)])

    return (
        train_generator,
        val_generator,
        train_generator0,
        val_generator0,
        train_generator1,
        val_generator1,
        train_generator2,
        val_generator2,
    )

def Inspect_data(df_train, num = 6, show_legend_every = 3):
    # SAMPLES
    # BATCH_SIZE which we are sure that they have mask for large_bowel
    Masks = list(df_train[df_train['large_bowel']!=''].sample(BATCH_SIZE).index)
    # BATCH_SIZE*2 which we are sure that they have mask for small_bowel
    Masks += list(df_train[df_train['small_bowel']!=''].sample(BATCH_SIZE*2).index)
    # BATCH_SIZE*3 which we are sure that they have mask for stomach
    Masks += list(df_train[df_train['stomach']!=''].sample(BATCH_SIZE*3).index)


    # DATA GENERATOR
    View_batches = DataGenerator(df_train[df_train.index.isin(Masks)],shuffle=True)

    # Visualizing
    fig = plt.figure(figsize=(10, 25), facecolor='slategrey')
    gs = gridspec.GridSpec(nrows=num, ncols=2)
    colors = ['red','green','blue']
    labels = ["Large Bowel", "Small Bowel", "Stomach"]
    patches = [ mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

    cmap1 = mpl.colors.ListedColormap(colors[0])
    cmap2 = mpl.colors.ListedColormap(colors[1])
    cmap3= mpl.colors.ListedColormap(colors[2])

    for i in range(num):
        images, mask = View_batches[i]
        sample_img=images[0,:,:,0]
        mask1=mask[0,:,:,0]
        mask2=mask[0,:,:,1]
        mask3=mask[0,:,:,2]
        
        ax0 = fig.add_subplot(gs[i, 0])
        im = ax0.imshow(sample_img, cmap='bone')

        ax1 = fig.add_subplot(gs[i, 1])
        if i%show_legend_every == 0:
            # ax0.set_title("Image", fontsize=15, weight='bold', y=1.02, color = 'black')
            # ax1.set_title("Mask", fontsize=15, weight='bold', y=1.02, color = 'black')
            plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4,fontsize = 14,title='Mask Labels', title_fontsize=14, edgecolor="black",  facecolor='black', )

        l0 = ax1.imshow(sample_img, cmap='bone')
        l1 = ax1.imshow(np.ma.masked_where(mask1== False,  mask1),cmap=cmap1, alpha=1)
        l2 = ax1.imshow(np.ma.masked_where(mask2== False,  mask2),cmap=cmap2, alpha=1)
        l3 = ax1.imshow(np.ma.masked_where(mask3== False,  mask3),cmap=cmap3, alpha=1)
        _ = [ax.set_axis_off() for ax in [ax0,ax1]]
        ax0.set_title("Image", fontsize=15, weight='bold', y=1.02, color= 'black')
        ax1.set_title("Mask", fontsize=15, weight='bold', y=1.02, color= 'black')
        colors = [im.cmap(im.norm(1)) for im in [l1,l2, l3]]
    return fig
        
    
    
    
def Plot_predicte_masks(model,df_train,valid_ids ,start = 20, end = 32, show_legend_every = 3):
    # start = 20
    # end = 32 
    pred_batches = DataGenerator(df_train[df_train.index.isin(valid_ids[start:end])],batch_size = 1,shuffle=True)
    preds = model.predict_generator(pred_batches,verbose=1)

    Threshold = 0.1
    # Visualizing
    fig = plt.figure(figsize=(10, 25), facecolor='slategrey')
    gs = gridspec.GridSpec(nrows=end-start, ncols=3)
    colors = ['yellow','green','red']
    labels = ["Large Bowel", "Small Bowel", "Stomach"]
    patches = [ mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

    cmap1 = mpl.colors.ListedColormap(colors[0])
    cmap2 = mpl.colors.ListedColormap(colors[1])
    cmap3= mpl.colors.ListedColormap(colors[2])


    for i in range(end-start):
        images, mask = pred_batches[i]
        sample_img=images[0,:,:,0]
        mask1=mask[0,:,:,0]
        mask2=mask[0,:,:,1]
        mask3=mask[0,:,:,2]
        
        pre=preds[i]
        predict1=pre[:,:,0]
        predict2=pre[:,:,1]
        predict3=pre[:,:,2]
        
        predict1= (predict1 > Threshold).astype(np.float32)
        predict2= (predict2 > Threshold).astype(np.float32)
        predict3= (predict3 > Threshold).astype(np.float32)
        
        ax0 = fig.add_subplot(gs[i, 0])
        im = ax0.imshow(sample_img, cmap='bone')
        ax0.set_title("Image", fontsize=12, y=1.01, color = 'black')
        #--------------------------
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.set_title("Mask", fontsize=12,  y=1.01, color = 'black')
        l0 = ax1.imshow(sample_img, cmap='bone')
        l1 = ax1.imshow(np.ma.masked_where(mask1== False,  mask1),cmap=cmap1, alpha=1)
        l2 = ax1.imshow(np.ma.masked_where(mask2== False,  mask2),cmap=cmap2, alpha=1)
        l3 = ax1.imshow(np.ma.masked_where(mask3== False,  mask3),cmap=cmap3, alpha=1)
        #--------------------------
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.set_title("Predict", fontsize=12, y=1.01, color = 'black')
        l0 = ax2.imshow(sample_img, cmap='bone')
        l1 = ax2.imshow(np.ma.masked_where(predict1== False,  predict1),cmap=cmap1, alpha=1)
        l2 = ax2.imshow(np.ma.masked_where(predict2== False,  predict2),cmap=cmap2, alpha=1)
        l3 = ax2.imshow(np.ma.masked_where(predict3== False,  predict3),cmap=cmap3, alpha=1)
        if i%show_legend_every == 0:
            plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4,fontsize = 12,title='Mask Labels', title_fontsize=12, edgecolor="black",  facecolor='black' )
        _ = [ax.set_axis_off() for ax in [ax0,ax1,ax2]]
        colors = [im.cmap(im.norm(1)) for im in [l1,l2, l3]]
    return fig
        
        
        