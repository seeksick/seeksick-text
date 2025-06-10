import pandas as pd

# 데이터 로드
df = pd.read_csv('data/train.csv')

# 감정별로 데이터 분리 및 저장
df_surprised = df[df['surprised'] == 1]
df_depressed = df[df['depressed'] == 1]
df_angry = df[df['angry'] == 1]

# 파일로 저장
df_surprised.to_csv('data/surprised_only.csv', index=False)
df_depressed.to_csv('data/depressed_only.csv', index=False)
df_angry.to_csv('data/angry_only.csv', index=False)

print('감정별 파일 저장 완료!') 