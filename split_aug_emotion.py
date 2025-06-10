import pandas as pd
from sklearn.utils import shuffle

# 파일 경로
files = {
    'depressed': 'data/test_ko_emotion_depressed.csv',
    'surprised': 'data/test_ko_emotion_surprised.csv',
    'angry': 'data/test_ko_emotion_angry.csv',
}

val_list = []
test_list = []
train_list = []

for emotion, file_path in files.items():
    df = pd.read_csv(file_path)
    df = shuffle(df, random_state=42).reset_index(drop=True)
    n = len(df)
    n_val = min(200, n // 2)
    n_test = min(200, n - n_val)
    val = df.iloc[:n_val]
    test = df.iloc[n_val:n_val+n_test]
    train = df.iloc[n_val+n_test:]
    val_list.append(val)
    test_list.append(test)
    train_list.append(train)

val_aug = pd.concat(val_list, ignore_index=True)
test_aug = pd.concat(test_list, ignore_index=True)
train_aug = pd.concat(train_list, ignore_index=True)

val_aug.to_csv('data/val_aug.csv', index=False)
test_aug.to_csv('data/test_aug.csv', index=False)
train_aug.to_csv('data/train_aug.csv', index=False)

print('val_aug, test_aug, train_aug 파일 저장 완료!') 