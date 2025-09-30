import math
import os
import pickle
import tarfile
import time

import cv2 as cv
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

# from config import device
device = torch.device("cpu")
from data_gen import data_transforms
from utils import align_face, get_central_face_attributes, get_all_face_attributes, draw_bboxes, ensure_folder

from face_model import MobileFaceNet, l2_norm, EdgeNeXt
from q_test import *

angles_file = 'data/angles.txt'
lfw_pickle = 'data/lfw_funneled.pkl'
transformer = data_transforms['val']

def match_keys_sequential(pretrained_dict, model_dict, print_info=False):
    """
    Match the keys of pretrained model and my model
    """
    # Check if the number of keys in pretrained model and my model are equal
    try:
        assert len(pretrained_dict) == len(model_dict)
    except AssertionError as DictLengthNotMatchError:
        print("The number of keys in pretrained model and my model are not equal.")
        return None
    
    # match the keys of pretrained model and my model
    print("Start matching keys...\n")
    for i, key in enumerate(pretrained_dict.keys()):
        if print_info:
            print(f'matching keys: {key} -> {list(model_dict.keys())[i]}')
        model_dict[list(model_dict.keys())[i]] = pretrained_dict[key]
    print("Matching keys done.\n")
    return model_dict

def extract(filename):
    with tarfile.open(filename, 'r') as tar:
        tar.extractall('data')


def process():
    subjects = [d for d in os.listdir('data/lfw_funneled') if os.path.isdir(os.path.join('data/lfw_funneled', d))]
    assert (len(subjects) == 5749), "Number of subjects is: {}!".format(len(subjects))

    print('Collecting file names...')
    file_names = []
    for i in tqdm(range(len(subjects))):
        sub = subjects[i]
        folder = os.path.join('data/lfw_funneled', sub)
        files = [f for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.jpg')]
        for file in files:
            filename = os.path.join(folder, file)
            file_names.append({'filename': filename, 'class_id': i, 'subject': sub})

    assert (len(file_names) == 13233), "Number of files is: {}!".format(len(file_names))

    print('Aligning faces...')
    samples = []
    for item in tqdm(file_names):
        filename = item['filename']
        class_id = item['class_id']
        sub = item['subject']
        is_valid, bounding_boxes, landmarks = get_central_face_attributes(filename)

        if is_valid:
            samples.append(
                {'class_id': class_id, 'subject': sub, 'full_path': filename, 'bounding_boxes': bounding_boxes,
                 'landmarks': landmarks})

    with open(lfw_pickle, 'wb') as file:
        save = {
            'samples': samples
        }
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)


def get_image(samples, file):
    filtered = [sample for sample in samples if file in sample['full_path'].replace('\\', '/')]
    assert (len(filtered) == 1), 'len(filtered): {} file:{}'.format(len(filtered), file)
    sample = filtered[0]
    # print(sample.keys())
    full_path = sample['full_path']
    landmarks = sample['landmarks']
    # print('full_path:', full_path, 'landmarks:', landmarks, "filtered:", filtered, len(filtered))
    img = align_face(full_path, landmarks)  # BGR
    return img


def transform(img, flip=False):
    if flip:
        img = cv.flip(img, 1)
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img


def get_feature(model, samples, file):
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float, device=device)
    img = get_image(samples, file)
    imgs[0] = transform(img.copy(), False)
    imgs[1] = transform(img.copy(), True)
    with torch.no_grad():
        output = model(imgs)
    feature_0 = output[0].cpu().numpy()
    feature_1 = output[1].cpu().numpy()
    feature = feature_0 + feature_1
    return feature / np.linalg.norm(feature)


def evaluate(model):
    model.eval()

    with open(lfw_pickle, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    filename = 'data/lfw_test_pair.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()

    angles = []

    elapsed = 0

    for line in tqdm(lines):
        temp = line.split('\t')

        if len(temp) == 3:
            tokens = [f'{temp[0]}/{temp[0]}_{int(temp[1]):04d}.jpg', 
            f'{temp[0]}/{temp[0]}_{int(temp[2]):04d}.jpg',
            1]
        elif len(temp) == 4:
            tokens = [f'{temp[0]}/{temp[0]}_{int(temp[1]):04d}.jpg', 
            f'{temp[2]}/{temp[2]}_{int(temp[3]):04d}.jpg',
            0]
        # print(tokens)

        start = time.time()
        x0 = get_feature(model, samples, tokens[0])
        x1 = get_feature(model, samples, tokens[1])
        end = time.time()
        elapsed += end - start

        cosine = np.dot(x0, x1)
        cosine = np.clip(cosine, -1.0, 1.0)
        theta = math.acos(cosine)
        theta = theta * 180 / math.pi
        is_same = tokens[2]
        angles.append('{} {}\n'.format(theta, is_same))

    print('elapsed: {} ms'.format(elapsed / (6000 * 2) * 1000))

    with open('data/angles.txt', 'w') as file:
        file.writelines(angles)


def visualize(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    ones = []
    zeros = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            ones.append(angle)
        else:
            zeros.append(angle)

    bins = np.linspace(0, 180, 181)

    plt.hist(zeros, bins, density=True, alpha=0.5, label='0', facecolor='red')
    plt.hist(ones, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu_0 = np.mean(zeros)
    sigma_0 = np.std(zeros)
    y_0 = scipy.stats.norm.pdf(bins, mu_0, sigma_0)
    plt.plot(bins, y_0, 'r--')
    mu_1 = np.mean(ones)
    sigma_1 = np.std(ones)
    y_1 = scipy.stats.norm.pdf(bins, mu_1, sigma_1)
    plt.plot(bins, y_1, 'b--')
    plt.xlabel('theta')
    plt.ylabel('theta j Distribution')
    plt.title(
        r'Histogram : mu_0={:.4f},sigma_0={:.4f}, mu_1={:.4f},sigma_1={:.4f}'.format(mu_0, sigma_0, mu_1, sigma_1))

    print('threshold: ' + str(threshold))
    print('mu_0: ' + str(mu_0))
    print('sigma_0: ' + str(sigma_0))
    print('mu_1: ' + str(mu_1))
    print('sigma_1: ' + str(sigma_1))

    plt.legend(loc='upper right')
    plt.plot([threshold, threshold], [0, 0.05], 'k-', lw=2)
    ensure_folder('images')
    plt.savefig('images/theta_dist.png')
    # plt.show()


def accuracy(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    wrong = 0
    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            if angle > threshold:
                wrong += 1
        else:
            if angle <= threshold:
                wrong += 1

    accuracy = 1 - wrong / 6000
    return accuracy


def show_bboxes(folder):
    with open(lfw_pickle, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']
    for sample in tqdm(samples):
        full_path = sample['full_path']
        bounding_boxes = sample['bounding_boxes']
        landmarks = sample['landmarks']
        img = cv.imread(full_path)
        img = draw_bboxes(img, bounding_boxes, landmarks)
        filename = os.path.basename(full_path)
        filename = os.path.join(folder, filename)
        cv.imwrite(filename, img)


def error_analysis(threshold):
    with open(angles_file) as file:
        angle_lines = file.readlines()

    fp = []
    fn = []
    for i, line in enumerate(angle_lines):
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if angle <= threshold and type == 0:
            fp.append(i)
        if angle > threshold and type == 1:
            fn.append(i)

    print('len(fp): ' + str(len(fp)))
    print('len(fn): ' + str(len(fn)))

    num_fp = len(fp)
    num_fn = len(fn)

    filename = 'data/lfw_test_pair.txt'
    with open(filename, 'r') as file:
        pair_lines = file.readlines()

    for i in range(num_fp):
        fp_id = fp[i]
        fp_line = pair_lines[fp_id]
        tokens = fp_line.split()
        file0 = tokens[0]
        copy_file(file0, '{}_fp_0.jpg'.format(i))
        save_aligned(file0, '{}_fp_0_aligned.jpg'.format(i))
        file1 = tokens[1]
        copy_file(file1, '{}_fp_1.jpg'.format(i))
        save_aligned(file1, '{}_fp_1_aligned.jpg'.format(i))

    for i in range(num_fn):
        fn_id = fn[i]
        fn_line = pair_lines[fn_id]
        tokens = fn_line.split()
        file0 = tokens[0]
        copy_file(file0, '{}_fn_0.jpg'.format(i))
        save_aligned(file0, '{}_fn_0_aligned.jpg'.format(i))
        file1 = tokens[1]
        copy_file(file1, '{}_fn_1.jpg'.format(i))
        save_aligned(file1, '{}_fn_1_aligned.jpg'.format(i))


def save_aligned(old_fn, new_fn):
    old_fn = os.path.join('data/lfw_funneled', old_fn)
    is_valid, bounding_boxes, landmarks = get_central_face_attributes(old_fn)
    img = align_face(old_fn, landmarks)
    new_fn = os.path.join('images', new_fn)
    cv.imwrite(new_fn, img)


def copy_file(old, new):
    old_fn = os.path.join('data/lfw_funneled', old)
    img = cv.imread(old_fn)
    bounding_boxes, landmarks = get_all_face_attributes(old_fn)
    draw_bboxes(img, bounding_boxes, landmarks)
    cv.resize(img, (224, 224))
    new_fn = os.path.join('images', new)
    cv.imwrite(new_fn, img)


def get_threshold():
    with open(angles_file, 'r') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'angle': angle, 'type': type})

    min_error = 6000
    min_threshold = 0

    for d in data:
        threshold = d['angle']
        type1 = len([s for s in data if s['angle'] <= threshold and s['type'] == 0])
        type2 = len([s for s in data if s['angle'] > threshold and s['type'] == 1])
        num_errors = type1 + type2
        if num_errors < min_error:
            min_error = num_errors
            min_threshold = threshold

    # print(min_error, min_threshold)
    return min_threshold


def lfw_test(model):
    filename = 'data/lfw-funneled.tgz'
    if not os.path.isdir('data/lfw_funneled'):
        print('Extracting {}...'.format(filename))
        extract(filename)

    # if not os.path.isfile(lfw_pickle):
    print('Processing {}...'.format(lfw_pickle))
    process()

    # if not os.path.isfile(angles_file):
    print('Evaluating {}...'.format(angles_file))
    evaluate(model)

    print('Calculating threshold...')
    # threshold = 70.36
    thres = get_threshold()
    print('Calculating accuracy...')
    acc = accuracy(thres)
    print('Accuracy: {}%, threshold: {}'.format(acc * 100, thres))
    return acc, thres


if __name__ == "__main__":
    # checkpoint = 'BEST_checkpoint.tar'
    # checkpoint = torch.load(checkpoint)
    # model = checkpoint['model'].module
    # model = model.to(device)
    # model.eval()

    model_name = 'yai_torchao'  
    if model_name is None:
        scripted_model_file = 'mobilefacenet_scripted.pt'
        model = torch.jit.load(scripted_model_file)
        model = model.to(device)
        model.eval()
    if model_name == 'mobilefacenet':
        model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
        model.load_state_dict(torch.load('Weights/MobileFace_Net', map_location=device))
        print('MobileFaceNet face detection model generated')
    elif model_name == 'pruned_yai':
        checkpoint_path = f'edgeface/checkpoints/pruned_yai_0_5.pt'
        model =  EdgeNeXt(depths=(2, 2, 6, 2), 
                                    dims=(24, 48, 88, 168), 
                                    heads=(4, 4, 4, 4),
                                    prune_ratio=0.5) # you can set prune_ratio to a value between 0 and 1 to load a pruned model
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
        print('Pruned EdgeFace detection model generated')

        model.eval().to(device)
    elif model_name == 'yai':
        checkpoint_path = f'edgeface/checkpoints/edgeface_xxs.pt'
        model =  EdgeNeXt(depths=(2, 2, 6, 2), dims=(24, 48, 88, 168), heads=(4, 4, 4, 4))
        
        model_state_dict = match_keys_sequential(torch.load(checkpoint_path, map_location='cpu'), model.state_dict(), print_info=False)
        model.load_state_dict(model_state_dict)
        model.to(device).eval()
        print('Yai model generated')
    elif model_name in ["yai_dynamic", "yai_ptq", "yai_qat", "yai_fp16", "yai_torchao"]:
        # 1) 먼저 pruned_yai_0_5 모델 불러오기
        checkpoint_path = "edgeface/checkpoints/pruned_yai_0_5.pt"
        original_model = EdgeNeXt(
            depths=(2, 2, 6, 2),
            dims=(24, 48, 88, 168),
            heads=(4, 4, 4, 4),
            prune_ratio=0.5
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        original_model.load_state_dict(state)
        print("Loaded pruned_yai_0_5 as base for quantization")

        # 2) 여기에 quantization 적용
        if model_name == "yai_dynamic":
            device = torch.device("cpu")
            model = quantize_edgenext_dynamic_advanced(copy.deepcopy(original_model)).to(device)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
        elif model_name == "yai_ptq":
            model = quantize_edgenext_ptq_advanced(copy.deepcopy(original_model), calib_loader, num_calibration_batches=500).to(device)
        elif model_name == "yai_qat":
            model = quantize_edgenext_qat_advanced(copy.deepcopy(original_model), train_loader=train_loader, num_epochs=3).to(device)
        elif model_name == "yai_fp16":
            model = quantize_edgenext_fp16(copy.deepcopy(original_model)).to(device)
        elif model_name == "yai_torchao":
            model = quantize_edgenext_torchao(copy.deepcopy(original_model)).to(device)

    acc, threshold = lfw_test(model)

    print('Visualizing {}...'.format(angles_file))
    visualize(threshold)

    print('error analysis...')
    error_analysis(threshold)
