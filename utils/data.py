import config

def load_data(path):
    if config.dataset == 'cifar100':
        dataset = load_dataset("cifar100")
        train_data = dataset['train']
        valid_data = dataset['test']
        train_img, train_label = train_data['img'], train_data['fine_label']
        valid_img, valid_label = valid_data['img'], valid_data['fine_label']
    elif config.dataset == 'tiny-imagenet':
        dataset = load_dataset("zh-plus/tiny-imagenet")
        train_data = dataset['train']
        valid_data = dataset['valid']
        train_img, train_label = train_data['image'], train_data['label']
        valid_img, valid_label = valid_data['image'], valid_data['label']