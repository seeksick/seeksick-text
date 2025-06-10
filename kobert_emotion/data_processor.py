# seeksick/seeksick-text/kobert_emotion/data_processor.py

import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split

def process_kemdy20_data(data_dir='../../KEMDy20_v1_2/annotation'):
    print(f"데이터 디렉토리: {os.path.abspath(data_dir)}")
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"찾은 CSV 파일 수: {len(csv_files)}")
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다. 경로: {data_dir}")

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, header=[0, 1])
            if not df.empty:
                print(f"파일 로드 성공: {file}, 행 수: {len(df)}")
                dfs.append(df)
            else:
                print(f"경고: 빈 파일 발견 - {file}")
        except Exception as e:
            print(f"파일 로드 실패: {file}, 에러: {str(e)}")

    if not dfs:
        raise ValueError("로드된 데이터프레임이 없습니다.")

    print(f"성공적으로 로드된 데이터프레임 수: {len(dfs)}")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"통합된 데이터프레임 크기: {combined_df.shape}")

    # 필요한 컬럼명 정의
    seg_col = ('Segment ID', ' ')
    eval_emotion_cols = [(f'Eval{str(i).zfill(2)}{"F" if i in [1,4,6,7,9] else "M"}', 'Emotion') for i in range(1, 11)]

    # 실제 컬럼명 확인
    print(f"실제 컬럼명 예시: {combined_df.columns.tolist()[:20]}")

    # 필요한 컬럼만 선택
    selected_df = combined_df[[seg_col] + eval_emotion_cols]
    selected_df.columns = ['Segment ID'] + [f'Eval{i}_Emotion' for i in range(1, 11)]

    # 소프트 레이블 생성 함수
    def create_soft_label(row):
        emotions = [str(row[f'Eval{i}_Emotion']).lower() for i in range(1, 11)]
        emotion_counts = {'happy': 0, 'depressed': 0, 'surprised': 0, 'angry': 0, 'neutral': 0}
        for emotion in emotions:
            if 'happy' in emotion:
                emotion_counts['happy'] += 1
            elif 'sad' in emotion:
                emotion_counts['depressed'] += 1
            elif 'surprise' in emotion:
                emotion_counts['surprised'] += 1
            elif 'angry' in emotion or 'disgust' in emotion:
                emotion_counts['angry'] += 1
            else:
                emotion_counts['neutral'] += 1
        total = sum(emotion_counts.values())
        if total == 0:
            return [0.2, 0.2, 0.2, 0.2, 0.2]
        return [emotion_counts[emo] / total for emo in ['happy', 'depressed', 'surprised', 'angry', 'neutral']]

    print("소프트 레이블 생성 중...")
    selected_df['soft_label'] = selected_df.apply(create_soft_label, axis=1)

    processed_df = pd.DataFrame({
        'text': selected_df['Segment ID'],
        'happy': [label[0] for label in selected_df['soft_label']],
        'depressed': [label[1] for label in selected_df['soft_label']],
        'surprised': [label[2] for label in selected_df['soft_label']],
        'angry': [label[3] for label in selected_df['soft_label']],
        'neutral': [label[4] for label in selected_df['soft_label']]
    })

    print(f"처리된 데이터프레임 크기: {processed_df.shape}")
    train_df, temp_df = train_test_split(processed_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    return train_df, val_df, test_df

if __name__ == "__main__":
    try:
        train_df, val_df, test_df = process_kemdy20_data()
        print("\n데이터 전처리 완료!")
        print(f"학습 데이터 크기: {len(train_df)}")
        print(f"검증 데이터 크기: {len(val_df)}")
        print(f"테스트 데이터 크기: {len(test_df)}")
    except Exception as e:
        print(f"\n에러 발생: {str(e)}")