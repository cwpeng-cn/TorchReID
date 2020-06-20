import zipfile


def download_and_prepare():
    reid_path = "/content/drive/My Drive/Colab/datasets/reid.zip"
    file_zip = zipfile.ZipFile(reid_path, 'r')
    for file in file_zip.namelist():
        file_zip.extract(file, r'.')

    with open("/content/drive/My Drive/Colab/ReID works/CVPR fintuning/resnet_ibn_b.py", "rb") as f, open(
            './resnet_ibn_b.py',
            'wb') as fw:
        fw.write(f.read())
    with open("/content/drive/My Drive/Colab/ReID works/CVPR fintuning/net_149.pth", "rb") as f, open('./net_149.pth',
                                                                                                      'wb') as fw:
        fw.write(f.read())


if __name__ == '__main__':
    download_and_prepare()
