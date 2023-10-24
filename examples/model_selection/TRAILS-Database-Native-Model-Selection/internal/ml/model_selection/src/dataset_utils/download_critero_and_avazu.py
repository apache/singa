# adapted from AFN-AAAI-20
import os
import zipfile
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if __name__ == "__main__":
    # if not os.path.exists('../data/avazu/'):
    #     os.mkdir('../data/avazu/')
    # print("Begin to download avazu data, the total size is 683MB...")
    # download('https://worksheets.codalab.org/rest/bundles/0xf5ab597052744680b1a55986557472c7/contents/blob/', '../data/avazu/avazu.zip')
    # print("Unzipping avazu dataset...")
    # with zipfile.ZipFile('../data/avazu/avazu.zip', 'r') as zip_ref:
    #     zip_ref.extractall('../data/avazu/')
    # print("Done.")

    if not os.path.exists('../exp_data/data/structure_data/criteo/'):
        os.mkdir('../exp_data/data/structure_data/criteo/')
    print("Begin to download criteo data, the total size is 3GB...")

    output_path = '../exp_data/data/structure_data/criteo/criteo.zip'
    if not os.path.exists(output_path):
        download('https://worksheets.codalab.org/rest/bundles/0x8dca5e7bac42470aa445f9a205d177c6/contents/blob/',
                 output_path)
    print("Unzipping criteo dataset...")
    with zipfile.ZipFile('../exp_data/data/structure_data/criteo/criteo.zip', 'r') as zip_ref:
        zip_ref.extractall('../exp_data/data/structure_data/criteo/')
    print("Done.")
