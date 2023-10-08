import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import random
import string
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据生成函数
def generate_data(num_samples):
    operators = ['+', '-', '*', '/']
    data = []
    for _ in range(num_samples):
        op = random.choice(operators)
        a, b = random.randint(0, 99), random.randint(1, 99)  # 避免除数为0
        expr = f'{a}{op}{b}'
        answer = str(eval(expr))
        data.append((expr, answer))
    return data

# 自定义 Dataset
class MathDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        expr, answer = self.data[idx]
        expr = torch.tensor([string.printable.index(c) for c in expr], dtype=torch.long)
        answer = torch.tensor([string.printable.index(c) for c in answer], dtype=torch.long)
        return expr, answer
# 数据准备
train_data = generate_data(10000)
test_data = generate_data(1000)
max_length = max(max(len(expr), len(answer)) for expr, answer in train_data + test_data)

MAX_LENGTH = 19  # 假设目标序列的最大长度是19

# def collate_fn(batch):
#     exprs, answers = zip(*batch)
#     # 将输入和目标序列填充到相同的固定长度
#     exprs = pad_sequence([torch.cat([e, torch.full((MAX_LENGTH - len(e),), len(string.printable) - 1, dtype=torch.long)]) for e in exprs], batch_first=True)
#     answers = pad_sequence([torch.cat([a, torch.full((MAX_LENGTH - len(a),), len(string.printable) - 1, dtype=torch.long)]) for a in answers], batch_first=True)
#     return exprs, answers
def collate_fn(batch):
    exprs, answers = zip(*batch)
    
    # 对于每个批次，动态确定 MAX_LENGTH
    max_length = max(max(len(expr), len(answer)) for expr, answer in batch)
    
    # 使用动态的 MAX_LENGTH 来填充序列
    exprs = pad_sequence([torch.cat([e, torch.full((max_length - len(e),), len(string.printable) - 1, dtype=torch.long)]) for e in exprs], batch_first=True)
    answers = pad_sequence([torch.cat([a, torch.full((max_length - len(a),), len(string.printable) - 1, dtype=torch.long)]) for a in answers], batch_first=True)
    
    return exprs, answers


# Transformer 模型定义
class SimpleMathTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(SimpleMathTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src, tgt = self.embedding(src), self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

train_dataset = MathDataset(train_data)
test_dataset = MathDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 模型实例化、优化器和损失函数
model = SimpleMathTransformer(
    vocab_size=len(string.printable),
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)
model = model.to(device)  # 将模型移动到GPU

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(10):
    model.train()
    for expr, answer in train_loader:
        expr, answer = expr.to(device), answer.to(device)  # 将数据移动到GPU
        optimizer.zero_grad()
        output = model(expr, expr)
        # # 打印调试信息
        # print(f'Output shape: {output.shape}')
        # print(f'Answer shape: {answer.shape}')
        loss = criterion(output.permute(0, 2, 1), answer)  # 现在损失计算应该使用匹配的形状

        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for expr, answer in test_loader:
            expr, answer = expr.to(device), answer.to(device)  # 确保数据也在相同的设备上
            output = model(expr, expr)
            _, predicted = output.max(2)
            total += answer.numel()
            correct += (predicted == answer).sum().item()
    print(f'Epoch {epoch + 1}, Accuracy: {correct / total * 100:.2f}%')

def eval_expression(expr):
    # 将输入表达式转换为张量
    expr_tensor = torch.tensor([string.printable.index(c) for c in expr], dtype=torch.long).unsqueeze(0)
    expr_tensor = expr_tensor.to(device) 
    # 使用模型进行预测
    with torch.no_grad():
        output = model(expr_tensor, expr_tensor)
    
    # 获取模型的输出，并将其转换回字符
    _, predicted_indices = output.max(2)
    predicted_chars = [string.printable[i] for i in predicted_indices[0]]
    result = ''.join(predicted_chars).strip()  # 移除任何额外的零或空格
    
    return result
# 在REPL中调用
result = eval_expression('45+67')
print(result)  # 输出模型的预测结果