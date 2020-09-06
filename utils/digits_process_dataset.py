import numpy as np
import time
import torch.nn.parallel
import torch.optim
import torch.utils.data
import scipy
import imageio
from scipy import io
import pickle
from utils.ops import *

def load_svhn(data_dir, split='train'):
    print('Loading SVHN dataset.')
    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'
    image_dir = os.path.join(data_dir, 'svhn', image_file)
    svhn = io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 255.
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels

def load_mnist(data_dir, split='train'):

    print('Loading MNIST dataset.')
    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(data_dir, 'mnist', image_file)
    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f, encoding="bytes")
    images = mnist[b'X']
    labels = mnist[b'y']
    images = images / 255.
    images = np.stack((images, images, images), axis=3)  # grayscale to rgb
    return np.squeeze(images[:10000]), labels[:10000]

def load_mnist_m(data_dir, split='train'):
    print('Loading MNIST_M dataset.')

    image_dir = os.path.join(data_dir, 'mnist_m')

    if split == 'train':
        data_dir = os.path.join(image_dir, 'mnist_m_train')
        with open(os.path.join(image_dir, 'mnist_m_train_labels.txt')) as f:
            content = f.readlines()

    elif split == 'test':
        data_dir = os.path.join(image_dir, 'mnist_m_test')
        with open(os.path.join(image_dir, 'mnist_m_test_labels.txt')) as f:
            content = f.readlines()

    content = [c.split('\n')[0] for c in content]
    images_files = [c.split(' ')[0] for c in content]
    labels = np.array([int(c.split(' ')[1]) for c in content]).reshape(-1)
    images = np.zeros((len(labels), 32, 32, 3))
    for no_img, img in enumerate(images_files):
        img_dir = os.path.join(data_dir, img)
        im = imageio.imread(img_dir)
        im = np.expand_dims(im, axis=0)
        images[no_img] = im
    images = images
    images = images / 255.
    return images, labels

def load_syn(data_dir, split='train'):
    print('Loading SYN dataset.')
    image_file = 'synth_train_32x32.mat' if split == 'train' else 'synth_test_32x32.mat'
    image_dir = os.path.join(data_dir, 'syn', image_file)
    syn = scipy.io.loadmat(image_dir)
    images = np.transpose(syn['X'], [3, 0, 1, 2])
    labels = syn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    images = images / 255.
    return images, labels

def load_usps(data_dir, split='train'):
    print('Loading USPS dataset.')
    image_file = 'usps_train_32x32.pkl' if split == 'train' else 'usps_test_32x32.pkl'
    # image_file = 'usps_32x32.pkl'
    image_dir = os.path.join(data_dir, 'usps', image_file)
    with open(image_dir, 'rb') as f:
        usps = pickle.load(f, encoding="bytes")
    images = usps['X']
    labels = usps['y']
    print('label range [{0}-{1}]'.format(np.min(labels), np.max(labels)))
    # labels -= 1
    # labels[labels == 255] = 9
    if np.max(images) == 255:
        images = images / 255.
    assert np.max(images) == 1
    images = np.squeeze(images)
    images = np.stack((images, images, images), axis=3)  # grayscale to rgb
    return images, labels

def load_test_data(data_dir, target, classes=None):

    if target == 'svhn':
        target_test_images, target_test_labels = load_svhn(data_dir, split='test')
    elif target == 'mnist':
        target_test_images, target_test_labels = load_mnist(data_dir, split='test')
    elif target == 'syn':
        target_test_images, target_test_labels = load_syn(data_dir, split='test')
    elif target == 'usps':
        target_test_images, target_test_labels = load_usps(data_dir, split='test')
    elif target == 'mnist_m':
        target_test_images, target_test_labels = load_mnist_m(data_dir, split='test')
    if classes:
        mask = np.isin(target_test_labels, list(classes))
        target_test_labels = target_test_labels[mask]
        target_test_images = target_test_images[mask]
    return target_test_images, target_test_labels

def asarray_and_reshape(imgs, labels):
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    imgs = np.reshape(imgs, (-1, 3, 32, 32))
    labels = np.reshape(labels, (-1,))
    return imgs, labels

def construct_datasets(data_dir, batch_size, kwargs, classes=None):

    def data2loader(imgs, labels):
        assert len(imgs) == len(labels)
        y = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # pytorch CHW, tf HWC
        X = torch.stack([torch.from_numpy(imgs[i]) for i in range(len(labels))])
        X_dataset = torch.utils.data.TensorDataset(X, y)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return X_loader

    train_imgs, train_labels = load_mnist(data_dir, 'train')
    val_imgs, val_labels = load_mnist(data_dir, 'test')
    if classes:
        mask = np.isin(train_labels, list(classes))
        train_imgs, train_labels = train_imgs[mask], train_labels[mask]
        mask = np.isin(val_labels, list(classes))
        val_imgs, val_labels = val_imgs[mask], val_labels[mask]
    return data2loader(train_imgs, train_labels), data2loader(val_imgs, val_labels)

def validate(val_loader, model):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    params = list(model.parameters())
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True).long()
        input = input.cuda(non_blocking=True).float()
        with torch.no_grad():
            output = model.functional(params, False, input)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg

def evaluation(model, data_dir, batch_size, kwargs):

    def data2loader(imgs, labels):
        assert len(imgs) == len(labels)
        y = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        X = torch.stack([torch.from_numpy(imgs[i]) for i in range(len(labels))])
        X_dataset = torch.utils.data.TensorDataset(X, y)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return  X_loader

    model.eval()
    params = list(model.parameters())
    accs = []
    target_domains = ['mnist', 'svhn', 'mnist_m', 'syn', 'usps']
    for td in target_domains:
        print(td)
        target_test_images, target_test_labels = load_test_data(data_dir, td)
        test_loader = data2loader(target_test_images, target_test_labels)

        batch_time = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        for i, (input, target) in enumerate(test_loader):
            target = target.cuda(non_blocking=True).long()
            input = input.cuda(non_blocking=True).float()
            with torch.no_grad():
                output = model.functional(params, False, input)
            # measure accuracy and record loss
            entropy = -torch.sum(F.softmax(output, 1) * F.log_softmax(output, 1), 1)
            target[target > 5] = 5
            unknown = ((entropy > 0.001).float() * (output.max() + 1e-8)).reshape(-1, 1)
            prec1 = accuracy(torch.cat([output, unknown], dim=1).data, target, topk=(1,))[0]
            # prec1 = accuracy(output.data, target, topk=(1,))[0]
            top1.update(prec1.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        accs.append(top1.avg)
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    avg_acc = np.mean(accs[1:])
    accs.append(avg_acc)
    print('avg acc', avg_acc)


def extract_features(model, data_dir, batch_size, kwargs):

    def data2loader(imgs, labels):
        assert len(imgs) == len(labels)
        y = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        X = torch.stack([torch.from_numpy(imgs[i]) for i in range(len(labels))])
        X_dataset = torch.utils.data.TensorDataset(X, y)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return  X_loader

    model.eval()
    params = list(model.parameters())
    accs = []
    target_domains = ['mnist', 'svhn', 'mnist_m', 'syn', 'usps']
    for td in target_domains:
        print(td)
        target_test_images, target_test_labels = load_test_data(data_dir, td)
        test_loader = data2loader(target_test_images, target_test_labels)

        batch_time = AverageMeter()
        end = time.time()
        features = []
        labels = []
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda(non_blocking=True).long()
            input = input.cuda(non_blocking=True).float()
            with torch.no_grad():
                feature, output = model.functional(params, False, input, return_feat=True)

            features.append(feature.cpu().numpy())
            labels.append(target.cpu().numpy())
            batch_time.update(time.time() - end)
            end = time.time()

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.save(f'known_{td}.npy', features[labels < 5])
        np.save(f'unknown_{td}.npy', features[labels >= 5])

def evaluation_old(model, data_dir, batch_size, kwargs):

    def data2loader(imgs, labels):
        assert len(imgs) == len(labels)
        y = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        X = torch.stack([torch.from_numpy(imgs[i]) for i in range(len(labels))])
        X_dataset = torch.utils.data.TensorDataset(X, y)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return  X_loader

    model.eval()
    params = list(model.parameters())
    accs = []
    target_domains = ['svhn', 'mnist_m', 'syn', 'usps']
    for td in target_domains:
        print(td)
        target_test_images, target_test_labels = load_test_data(data_dir, td)
        test_loader = data2loader(target_test_images, target_test_labels)

        batch_time = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        for i, (input, target) in enumerate(test_loader):
            target = target.cuda(non_blocking=True).long()
            input = input.cuda(non_blocking=True).float()
            with torch.no_grad():
                output = model.functional(params, False, input)
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            top1.update(prec1.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        accs.append(top1.avg)
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    avg_acc = np.mean(accs[1:])
    accs.append(avg_acc)
    print('avg acc', avg_acc)

def visualize(model, data_dir, batch_size, kwargs):
    def data2loader(imgs, labels):
        assert len(imgs) == len(labels)
        y = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        X = torch.stack([torch.from_numpy(imgs[i]) for i in range(len(labels))])
        X_dataset = torch.utils.data.TensorDataset(X, y)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return  X_loader

    model.eval()
    params = list(model.parameters())    
    #target_domains = ['mnist', 'svhn', 'mnist_m', 'syn', 'usps']
    target_domains = ['mnist','mnist_m', 'usps']
    outputs = []
    labels = []
    flags = []
    corrects = []
    entropies = []
    for td in target_domains:
        print(td)
        target_test_images, target_test_labels = load_test_data(data_dir, td)
        test_loader = data2loader(target_test_images, target_test_labels)

        for i, (input, target) in enumerate(test_loader):
            input = input.cuda(non_blocking=True).float()
            with torch.no_grad():
                feature, output = model.functional(params, False, input, return_feat=True)

            correct = (output.cpu().argmax(dim=1) == target.cpu().long()).numpy()
            entropy = -torch.sum(F.softmax(output, 1) * F.log_softmax(output, 1), 1).cpu().numpy()
            corrects.append(correct)
            outputs.append(feature.cpu().numpy())
            labels.append(target.numpy())
            flag = np.ones(batch_size)
            mask = np.isin(target.numpy(), [0, 1, 2, 3, 4])
            if td == 'mnist':
                flag = np.where(mask, 0, 1)
            else:
                flag = np.where(mask, 2, 3)
            flags.append(flag)
            entropies.append(entropy)
    outputs = np.concatenate(outputs, axis=0)[::10]
    labels = np.concatenate(labels, axis=0)[::10]
    flags = np.concatenate(flags, axis=0)[::10]
    corrects = np.concatenate(corrects, axis=0)[::10]
    entropies = np.concatenate(entropies, axis=0)[::10]
    import umap
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    embedding = umap.UMAP().fit_transform(outputs)
    cmap = plt.get_cmap("tab10")
    markers = ['o', 'x', '^', 's']
    plt.figure(figsize=(20, 15))
    for flag in [0, 2]:
        idx = (flags == flag) & (corrects == 1) & (labels < 5)
        print(flag, 'known class correct', entropies[idx].mean())
        plt.scatter(embedding[idx, 0], embedding[idx, 1], c=labels[idx], marker=markers[flag], alpha=0.7)
        plt.clim(0, 10)
        st = 'source' if flag == 0 else 'target'
        # plt.hist(entropies[idx], alpha=0.5, label=f'{st} known correct', bins=50)
        idx = (flags == flag) & (labels < 5) & (corrects != 1)
        print(flag, 'known class don\'t correct', entropies[idx].mean())
        # plt.hist(entropies[idx], alpha=0.5, label=f'{st} known uncorrect', bins=50)
        plt.scatter(embedding[idx, 0], embedding[idx, 1], c=np.ones_like(labels[idx])*10, marker=markers[flag], alpha=0.7)
        plt.clim(0, 10)
    for flag in [1, 3]:
        idx = (flags == flag) & (labels >= 5)
        print(flag, 'unknown class', entropies[idx].mean())
        st = 'source' if flag == 1 else 'target'
        # plt.hist(entropies[idx], alpha=0.5, label=f'{st} unknown', bins=50)
        plt.scatter(embedding[idx, 0], embedding[idx, 1], c=labels[idx], marker=markers[flag], alpha=0.7)

    plt.clim(labels.min(), labels.max()+1)
    plt.colorbar()
    # plt.legend()
    plt.savefig('fig.png')
    
