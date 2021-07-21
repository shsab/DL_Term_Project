from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from timegan import TimeGAN

import os
import requests as req
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def real_data_loading(data: np.array, seq_len):
    # Flip the data to make chronological data
    ori_data = data[::-1]
    # Normalize the data
    scaler = MinMaxScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data


def get_data(seq_len: int):
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'stock.csv')
    try:
        stock_df = pd.read_csv(file_path)
    except:
        stock_url = 'https://query1.finance.yahoo.com/v7/finance/download/GOOG?period1=1483228800&period2=1611446400&interval=1d&events=history&includeAdjustedClose=true'
        request = req.get(stock_url)
        url_content = request.content

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        stock_csv = open(file_path, "wb")
        stock_csv.write(url_content)
        # Reading the stock data
        stock_df = pd.read_csv(file_path)

    try:
        stock_df = stock_df.set_index('Date').sort_index()
    except:
        stock_df=stock_df
    #Data transformations to be applied prior to be used with the synthesizer model
    processed_data = real_data_loading(stock_df.values, seq_len=seq_len)

    return processed_data


if __name__ == '__main__':
    # Specific to TimeGANs
    seq_len = 24
    n_seq = 6
    hidden_dim = 24
    gamma = 1
    noise_dim = 32
    dim = 128
    batch_size = 128
    log_step = 100
    learning_rate = 5e-4
    gan_args = [batch_size, learning_rate, noise_dim, 24, 2, (0, 1), dim]

    stock_data = get_data(seq_len=seq_len)
    print(len(stock_data), stock_data[0].shape)

    if path.exists(os.path.join(os.path.dirname(__file__), 'model', 'synthesizer_stock.pkl')):
        synthysizer = TimeGAN.load(os.path.join(os.path.dirname(__file__), 'model', 'synthesizer_stock.pkl'))
    else:
        synthysizer = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
        synthysizer.train(stock_data, train_steps=200)
        # synth.save(os.path.join(os.path.dirname(__file__), 'model', 'synthesizer_stock.pkl'))

    synth_data = synthysizer.sample(len(stock_data))
    print(synth_data.shape)
    try:
        np.savetxt(os.path.join(os.path.dirname(__file__), 'data', 'synthesized_stock.csv'), synth_data, delimiter=',')
    except Exception:
        print('Saving synthesized_stock failed!')
        pass

    try:
        pd.DataFrame(synth_data).to_csv(os.path.join(os.path.dirname(__file__), 'data', 'synthesized_stock_df.csv'))
    except Exception:
        print('Saving synthesized_stock failed!')
        pass

    # Reshaping the data
    cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    axes = axes.flatten()

    time = list(range(1, 25))
    obs = np.random.randint(len(stock_data))

    for j, col in enumerate(cols):
        df = pd.DataFrame({'Real': stock_data[obs][:, j],
                           'Synthetic': synth_data[obs][:, j]})
        df.plot(ax=axes[j],
                title=col,
                secondary_y='Synthetic data', style=['-', '--'])
    fig.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'graphs', 'data_comparison.png'))

    sample_size = 250
    idx = np.random.permutation(len(stock_data))[:sample_size]

    real_sample = np.asarray(stock_data)[idx]
    synthetic_sample = np.asarray(synth_data)[idx]

    # for the purpose of comparision we need the data to be 2-Dimensional. For that reason we are going to use only two componentes for both the PCA and TSNE.
    synth_data_reduced = real_sample.reshape(-1, seq_len)
    stock_data_reduced = np.asarray(synthetic_sample).reshape(-1, seq_len)

    n_components = 2
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components, n_iter=300)

    # The fit of the methods must be done only using the real sequential data
    pca.fit(stock_data_reduced)

    pca_real = pd.DataFrame(pca.transform(stock_data_reduced))
    pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

    data_reduced = np.concatenate((stock_data_reduced, synth_data_reduced), axis=0)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

    # The scatter plots for PCA and TSNE methods
    import matplotlib.gridspec as gridspec

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    # TSNE scatter plot
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title('PCA results',
                 fontsize=20,
                 color='red',
                 pad=10)

    # PCA scatter plot
    plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values,
                c='black', alpha=0.2, label='Original')
    plt.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1],
                c='red', alpha=0.2, label='Synthetic')
    ax.legend()

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title('TSNE results',
                  fontsize=20,
                  color='red',
                  pad=10)

    plt.scatter(tsne_results.iloc[:sample_size, 0].values, tsne_results.iloc[:sample_size, 1].values,
                c='black', alpha=0.2, label='Original')
    plt.scatter(tsne_results.iloc[sample_size:, 0], tsne_results.iloc[sample_size:, 1],
                c='red', alpha=0.2, label='Synthetic')

    ax2.legend()

    fig.suptitle('Validating synthetic vs real data diversity and distributions',
                 fontsize=16,
                 color='grey')

    # fig.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'graphs', 'synthetic_vs_real.png'))
