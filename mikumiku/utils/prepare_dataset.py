"""
used for mini-dataset prepare on local environment
"""

from dataset_prepare.argoverse_v2_dataset import ArgoverseV2Dataset

def prepare_data(root,  
                 train_raw_dir, 
                 val_raw_dir,
                 test_raw_dir,
                 train_processed_dir, 
                 val_processed_dir,
                 test_processed_dir,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None) -> None:
        ArgoverseV2Dataset(root, 'train', train_raw_dir, train_processed_dir, train_transform)
        ArgoverseV2Dataset(root, 'val', val_raw_dir, val_processed_dir, val_transform)
        ArgoverseV2Dataset(root, 'test', test_raw_dir, test_processed_dir, test_transform)
if __name__ =='__main__':
    prepare_data(
          root = 'D:\\argoverse2',
          train_raw_dir='D:\\argoverse2\\train\\raw',
          val_raw_dir='D:\\argoverse2\\val\\raw',
          test_raw_dir='D:\\argoverse2\\test\\raw',
          train_processed_dir='D:\\argoverse2\\train\\processed',
          val_processed_dir='D:\\argoverse2\\val\\processed',
          test_processed_dir='D:\\argoverse2\\test\\processed',
    )