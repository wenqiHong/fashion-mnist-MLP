import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
import gzip
import urllib.request

def load_fashion_mnist(data_dir='./data/fashion'):
    os.makedirs(data_dir, exist_ok=True)
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    
    def download(fname):
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, fpath)
        return fpath
    
    def read_images(path):
        with gzip.open(path, 'rb') as f:
            _ = int.from_bytes(f.read(4), 'big')
            n = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            buf = f.read(n * rows * cols)
            return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows*cols).astype(np.float32) / 255.0
    
    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            _ = int.from_bytes(f.read(4), 'big')
            n = int.from_bytes(f.read(4), 'big')
            return np.frombuffer(f.read(n), dtype=np.uint8)
    
    X_full = read_images(download(files['train_images']))
    y_full = read_labels(download(files['train_labels']))
    X_test = read_images(download(files['test_images']))
    y_test = read_labels(download(files['test_labels']))
    
    x_val, y_val = X_full[-5000:], y_full[-5000:]
    x_train, y_train = X_full[:-5000], y_full[:-5000]
    
    print(f"训练集: {x_train.shape}, 验证集: {x_val.shape}, 测试集: {X_test.shape}")
    return x_train, y_train, x_val, y_val, X_test, y_test


# ===================== 2. 激活函数与导数 =====================
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


# ===================== 3. 三层神经网络（两个隐藏层，输入层不计）=====================
class ThreeLayerNet:
    """三个可训练层: 隐藏层1、隐藏层2、输出层"""
    def __init__(self, input_size=784, hidden1_size=256, hidden2_size=128, output_size=10, activation='relu'):
        self.activation = activation
        # Xavier 初始化
        self.params = {
            'W1': np.random.randn(input_size, hidden1_size) / np.sqrt(input_size),
            'b1': np.zeros(hidden1_size),
            'W2': np.random.randn(hidden1_size, hidden2_size) / np.sqrt(hidden1_size),
            'b2': np.zeros(hidden2_size),
            'W3': np.random.randn(hidden2_size, output_size) / np.sqrt(hidden2_size),
            'b3': np.zeros(output_size)
        }
        self.cache = {}

    def forward(self, x):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # 第一隐藏层
        z1 = x @ W1 + b1
        a1 = relu(z1) if self.activation == 'relu' else sigmoid(z1)
        grad1 = relu_grad(z1) if self.activation == 'relu' else sigmoid_grad(z1)

        # 第二隐藏层
        z2 = a1 @ W2 + b2
        a2 = relu(z2) if self.activation == 'relu' else sigmoid(z2)
        grad2 = relu_grad(z2) if self.activation == 'relu' else sigmoid_grad(z2)

        # 输出层
        z3 = a2 @ W3 + b3
        exp_z = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        y_pred = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # 缓存中间结果
        self.cache.update({'x': x, 'a1': a1, 'a2': a2, 'grad1': grad1, 'grad2': grad2, 'y_pred': y_pred})
        return y_pred

    def backward(self, y_true, reg_lambda):
        x = self.cache['x']
        a1 = self.cache['a1']
        a2 = self.cache['a2']
        grad1 = self.cache['grad1']
        grad2 = self.cache['grad2']
        y_pred = self.cache['y_pred']
        W2 = self.params['W2']
        W3 = self.params['W3']
        batch_size = x.shape[0]

        # 输出层梯度
        dz3 = y_pred.copy()
        dz3[range(batch_size), y_true] -= 1
        dz3 /= batch_size

        dW3 = a2.T @ dz3 + reg_lambda * W3
        db3 = np.sum(dz3, axis=0)

        # 第二隐藏层梯度
        da2 = dz3 @ W3.T
        dz2 = da2 * grad2
        dW2 = a1.T @ dz2 + reg_lambda * W2
        db2 = np.sum(dz2, axis=0)

        # 第一隐藏层梯度
        da1 = dz2 @ W2.T
        dz1 = da1 * grad1
        dW1 = x.T @ dz1 + reg_lambda * self.params['W1']
        db1 = np.sum(dz1, axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

def cross_entropy_loss(y_pred, y_true, reg_lambda, params):
    loss = -np.mean(np.log(y_pred[range(len(y_true)), y_true] + 1e-8))
    l2 = 0.5 * reg_lambda * (np.sum(params['W1']**2) + np.sum(params['W2']**2) + np.sum(params['W3']**2))
    return loss + l2

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == y_true)

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model.params, f)

def load_model(model, path):
    with open(path, 'rb') as f:
        model.params = pickle.load(f)


def train_net(x_train, y_train, x_val, y_val,
              hidden1_size=256, hidden2_size=128, activation='relu',
              lr=0.1, reg_lambda=1e-4, epochs=50, batch_size=128, lr_decay=0.95):
    """训练函数"""
    model = ThreeLayerNet(input_size=784, hidden1_size=hidden1_size, hidden2_size=hidden2_size,
                          output_size=10, activation=activation)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_params = None

    for epoch in range(epochs):
        idx = np.random.permutation(len(x_train))
        x_s, y_s = x_train[idx], y_train[idx]

        for i in range(0, len(x_train), batch_size):
            x_batch = x_s[i:i+batch_size]
            y_batch = y_s[i:i+batch_size]
            y_pred = model.forward(x_batch)
            grads = model.backward(y_batch, reg_lambda)
            for key in model.params:
                model.params[key] -= lr * grads[f'd{key}']

        lr *= lr_decay   # 学习率衰减

        y_train_pred = model.forward(x_train)
        y_val_pred = model.forward(x_val)
        train_loss = cross_entropy_loss(y_train_pred, y_train, reg_lambda, model.params)
        val_loss = cross_entropy_loss(y_val_pred, y_val, reg_lambda, model.params)
        train_acc = accuracy(y_train_pred, y_train)
        val_acc = accuracy(y_val_pred, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_val_acc:.4f}")

    model.params = best_params
    return model, train_losses, val_losses, train_accs, val_accs


# ===================== 6. 超参数网格搜索（作业要求必备）=====================
def hyperparameter_search(x_train, y_train, x_val, y_val):
    """超参数查找"""
    param_grid = {
        'lr': [0.05, 0.1, 0.2],
        'hidden1_size': [128, 256],
        'hidden2_size': [64, 128],
        'reg_lambda': [1e-4, 5e-4]
    }

    best_overall_acc = 0
    best_params = None
    results = []

    print("\n========== 开始超参数网格搜索 ==========")
    # 遍历所有超参数组合
    for lr in param_grid['lr']:
        for h1 in param_grid['hidden1_size']:
            for h2 in param_grid['hidden2_size']:
                for reg in param_grid['reg_lambda']:
                    print(f"\n训练组合: lr={lr}, h1={h1}, h2={h2}, reg={reg}")
                    model, _, _, _, val_accs = train_net(
                        x_train, y_train, x_val, y_val,
                        hidden1_size=h1, hidden2_size=h2, activation="relu",
                        lr=lr, reg_lambda=reg, epochs=10
                    )
                    current_val_acc = max(val_accs)
                    results.append({
                        'lr': lr, 'h1': h1, 'h2': h2, 'reg': reg,  'val_acc': current_val_acc
                    })
                    if current_val_acc > best_overall_acc:
                        best_overall_acc = current_val_acc
                        best_params = (lr, h1, h2, reg)
                    print(f"当前组合最高验证精度: {current_val_acc:.4f}")

    print("\n========== 超参数搜索完成 ==========")
    print(f"最佳组合: lr={best_params[0]}, h1={best_params[1]}, h2={best_params[2]}, reg={best_params[3]}")
    print(f"最佳验证精度: {best_overall_acc:.4f}")
    return best_params, results

def plot_curves(train_losses, val_losses, train_accs, val_accs):
    """损失曲线和准确率曲线"""
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, x_test, y_test):
    """混淆矩阵"""
    y_pred = np.argmax(model.forward(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def visualize_weights(model, layer='W1'):
    """可视化第一隐藏层权重（展成28x28）"""
    W = model.params['W1'].T
    plt.figure(figsize=(10,10))
    for i in range(min(25, W.shape[0])):
        plt.subplot(5,5,i+1)
        plt.imshow(W[i].reshape(28,28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'{layer} Weights Visualization')
    plt.show()


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = load_fashion_mnist()

    best_params, results = hyperparameter_search(x_train, y_train, x_val, y_val)        #超参数查找
    lr_best, h1_best, h2_best, reg_best = best_params

    print("\n========== 使用最佳参数正式训练 ==========")
    model, train_loss, val_loss, train_acc, val_acc = train_net(
        x_train, y_train, x_val, y_val,
        hidden1_size=h1_best, hidden2_size=h2_best, activation="relu",
        lr=lr_best, reg_lambda=reg_best, epochs=30
    )

    save_model(model, 'three_layer_net_best.pkl')
    print("\n最优模型已保存至 three_layer_net_best.pkl")

    test_pred = model.forward(x_test)
    test_acc = accuracy(test_pred, y_test)
    print(f"\n测试集最终准确率: {test_acc:.4f}")

    plot_curves(train_loss, val_loss, train_acc, val_acc)
    plot_confusion_matrix(model, x_test, y_test)
    visualize_weights(model, layer='W1')