import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# 기존 데이터 로드
df_train = pd.read_csv('data/train.csv')
df_val = pd.read_csv('data/val.csv')
df_test = pd.read_csv('data/test.csv')

# 기존 데이터의 text set 생성 (중복 제거용)
train_texts = set(df_train['text'])
val_texts = set(df_val['text'])
test_texts = set(df_test['text'])
all_texts = train_texts | val_texts | test_texts

# 증강 데이터 로드
df_aug = pd.read_csv('data/test_ko_emotion.csv')

# 기존 데이터와 중복되지 않는 증강 데이터만 추출
aug_unique = df_aug[~df_aug['text'].isin(all_texts)].reset_index(drop=True)

# 기존 데이터 개수로 비율 계산
total = len(df_train) + len(df_val) + len(df_test)
train_ratio = len(df_train) / total
val_ratio = len(df_val) / total
test_ratio = len(df_test) / total

# 증강 데이터 분할
df_temp, df_test_aug = train_test_split(aug_unique, test_size=test_ratio, random_state=42)
df_train_aug, df_val_aug = train_test_split(df_temp, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)

# 파일로 저장
df_train_aug.to_csv('data/train_aug.csv', index=False)
df_val_aug.to_csv('data/val_aug.csv', index=False)
df_test_aug.to_csv('data/test_aug.csv', index=False)

print('증강 데이터 분할 및 저장 완료!')

# 데이터 로드
val_aug = pd.read_csv('data/val_aug.csv')
test_aug = pd.read_csv('data/test_aug.csv')
train_aug = pd.read_csv('data/train_aug.csv')
angry = pd.read_csv('data/Angry_Emotion_Dataset.csv')
depressed = pd.read_csv('data/Depressed_Emotion_Dataset.csv')

# angry, depressed 합치기
new_aug = pd.concat([angry, depressed], ignore_index=True)

# 감정별로 분리
emotion_names = ['happy', 'depressed', 'surprised', 'angry', 'neutral']

# 각 감정별 최소 샘플 개수 구하기 (최대 30개, 부족하면 가능한 만큼)
n_per_emotion = min(
    int(new_aug[emotion_names].sum().min()),
    30
)

val_samples = []
test_samples = []

for emo in emotion_names:
    emo_df = new_aug[new_aug[emo] == 1]
    emo_df = shuffle(emo_df, random_state=42)
    # val/test에 반반씩 분배
    half = n_per_emotion // 2
    val_samples.append(emo_df.iloc[:half])
    test_samples.append(emo_df.iloc[half:n_per_emotion])

val_new = pd.concat(val_samples, ignore_index=True)
test_new = pd.concat(test_samples, ignore_index=True)

# val/test에 기존 aug append
val_final = pd.concat([val_new, val_aug], ignore_index=True)
test_final = pd.concat([test_new, test_aug], ignore_index=True)

# train_aug는 나머지
used_texts = set(val_final['text']) | set(test_final['text'])
train_new = new_aug[~new_aug['text'].isin(used_texts)]
train_final = pd.concat([train_aug, train_new], ignore_index=True)

# 저장
val_final.to_csv('data/val_aug.csv', index=False)
test_final.to_csv('data/test_aug.csv', index=False)
train_final.to_csv('data/train_aug.csv', index=False)
print('균등 샘플링 및 분할 완료!')

# 데이터 로드
df = pd.read_csv('data/test_ko_emotion.csv')

# angry, surprised, depressed만 분배
emo_targets = ['angry', 'surprised', 'depressed']
val_aug_list = []
test_aug_list = []
used_idx = set()

for emo in emo_targets:
    emo_df = df[df[emo] == 1]
    emo_df = shuffle(emo_df, random_state=42)
    n = min(200, len(emo_df))
    val_aug_list.append(emo_df.iloc[:n//2])
    test_aug_list.append(emo_df.iloc[n//2:n])
    used_idx.update(emo_df.index[:n])

val_aug = pd.concat(val_aug_list, ignore_index=True)
test_aug = pd.concat(test_aug_list, ignore_index=True)

# 남은 angry, surprised, depressed 샘플 + happy, neutral 등 나머지는 train_aug
remain_idx = set(df.index) - used_idx
train_aug = df.loc[list(remain_idx)].copy()

val_aug.to_csv('data/val_aug.csv', index=False)
test_aug.to_csv('data/test_aug.csv', index=False)
train_aug.to_csv('data/train_aug.csv', index=False)
print('정확히 분배 완료!')

df = pd.read_csv('data/test_ko_emotion.csv')

emotions = ['angry', 'surprised', 'depressed', 'happy', 'neutral']
for emo in emotions:
    emo_df = df[df[emo] == 1]
    emo_df.to_csv(f'data/{emo}.csv', index=False)
print('감정별 csv 저장 완료!') 