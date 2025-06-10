import pandas as pd

# 입력 및 출력 파일 경로
input_path = 'data/raw/korean_emotion_aihub.csv'
output_path = 'data/new_test_aug.csv'

# 변환할 감정 라벨 매핑
emotion_map = {
    '행복': 'happy',
    '슬픔': 'depressed',
    '놀람': 'surprised',
    '분노': 'angry',
    '중립': 'neutral',
}

# 사용할 감정 리스트 (공포, 혐오 제외)
target_emotions = ['happy', 'depressed', 'surprised', 'angry', 'neutral']

# 데이터 읽기
df = pd.read_csv(input_path)

# 변환 결과 저장용 리스트
rows = []

for idx, row in df.iterrows():
    text = row['Sentence']
    emotion = row['Emotion'].strip()
    # 공포, 혐오 제외
    if emotion in ['공포', '혐오']:
        continue
    # 감정 매핑이 없는 경우 스킵
    if emotion not in emotion_map:
        continue
    # 원핫 인코딩
    onehot = [0] * len(target_emotions)
    onehot[target_emotions.index(emotion_map[emotion])] = 1
    rows.append([text] + onehot)

# 데이터프레임 생성 및 저장
out_df = pd.DataFrame(rows, columns=['text'] + target_emotions)
out_df.to_csv(output_path, index=False)

print(f"변환 완료! 저장 위치: {output_path}") 