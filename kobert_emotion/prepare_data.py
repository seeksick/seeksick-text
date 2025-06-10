import os
import pandas as pd
import glob
from tqdm import tqdm
import logging
import re
from sklearn.model_selection import train_test_split
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터셋 기본 경로 설정
BASE_DATASET_PATH = r'C:\Users\Server1\seeksick\KEMDy20_v1_2'

def read_text_file(file_path):
    """텍스트 파일을 읽고 전처리합니다."""
    try:
        with open(file_path, 'r', encoding='euc-kr') as f:
            text = f.read().strip()
            # 특수기호 패턴(n/, b/, u/, c/ 등) 제거
            text = re.sub(r"\b[nbuc]/", "", text)
            # 따옴표 제거
            text = text.replace('"', '').replace("'", "")
            return text
    except Exception as e:
        logger.error(f"Failed to read file: {file_path} ({e})")
        return None

def prepare_dataset():
    """KEMDy20 데이터셋을 준비합니다."""
    annotation_files = glob.glob(os.path.join(BASE_DATASET_PATH, 'annotation', '*_eval.csv'))
    logger.info(f"Found {len(annotation_files)} annotation files")
    
    all_data = []
    not_found_files = []
    emotion_columns = ['happy', 'depressed', 'surprised', 'angry', 'neutral']
    
    for annotation_file in tqdm(annotation_files, desc="Processing annotation files"):
        try:
            # 세션 번호 추출 (예: Sess01 -> 01)
            session_name = os.path.basename(annotation_file).split('_')[0]
            session_num = session_name.replace('Sess', '')
            session_folder = f'Session{session_num}'
            wav_dir = os.path.join(BASE_DATASET_PATH, 'wav', session_folder)
            
            # annotation 파일 읽기
            session_data = pd.read_csv(annotation_file, header=[0,1])
            logger.info(f"Processing session: {session_name}")
            
            for _, row in session_data.iterrows():
                try:
                    segment_id = row[('Segment ID', ' ')]
                    if pd.isna(segment_id):
                        continue
                    # 텍스트 파일 경로: KEMDy20_v1_2/wav/Session{숫자}/{SegmentID}.txt
                    text_file = os.path.join(wav_dir, f'{segment_id}.txt')
                    if not os.path.exists(text_file):
                        if len(not_found_files) < 10:
                            not_found_files.append(text_file)
                        continue
                    # 텍스트 읽기
                    text = read_text_file(text_file)
                    if text is None or text.strip() == '':
                        continue
                    # 감정 레이블 수집
                    emotions = []
                    for i in range(1, 11):
                        eval_col = (f'Eval{i:02d}{"F" if i % 2 == 1 else "M"}', 'Emotion')
                        try:
                            emotion = row[eval_col]
                            if pd.notna(emotion):
                                emotion = str(emotion).strip().lower()
                                # 감정 매핑
                                if 'sad' in emotion or 'depressed' in emotion or '슬픔' in emotion or '우울' in emotion:
                                    emotions.append('depressed')
                                elif 'happy' in emotion or 'joy' in emotion or '기쁨' in emotion or '행복' in emotion:
                                    emotions.append('happy')
                                elif 'surprised' in emotion or '놀람' in emotion or '놀라움' in emotion:
                                    emotions.append('surprised')
                                elif 'angry' in emotion or '화남' in emotion or '분노' in emotion:
                                    emotions.append('angry')
                                elif 'neutral' in emotion or '무감정' in emotion:
                                    emotions.append('neutral')
                        except KeyError:
                            continue
                    if len(emotions) == 0:
                        continue
                    # 최빈값(one-hot) 레이블 생성
                    counts = Counter(emotions)
                    main_emotion = counts.most_common(1)[0][0]
                    label = {emo: 1 if emo == main_emotion else 0 for emo in emotion_columns}
                    all_data.append({
                        'text': text,
                        **label
                    })
                except Exception as e:
                    logger.error(f"Error processing row: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error processing file {annotation_file}: {str(e)}")
            continue
    if not_found_files:
        logger.info(f"존재하지 않는 파일 예시: {not_found_files}")
    df = pd.DataFrame(all_data)
    logger.info(f"Total samples collected: {len(df)}")
    if len(df) == 0:
        logger.error("No data was collected!")
        return
    # train/val/test 분할
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    # neutral 데이터 비중 조절 (train 데이터만)
    neutral_mask = (train_df['neutral'] == 1) & (train_df[['happy', 'depressed', 'surprised', 'angry']].sum(axis=1) == 0)
    neutral_positive = train_df[neutral_mask].sample(frac=0.5, random_state=42)
    non_neutral = train_df[~neutral_mask]
    train_df = pd.concat([neutral_positive, non_neutral])
    train_df = train_df.sample(frac=1, random_state=42)
    # 각 감정별 positive 샘플 수 로깅
    for emotion in emotion_columns:
        positive_count = len(train_df[train_df[emotion] == 1])
        logger.info(f"{emotion} positive samples: {positive_count}")
    # CSV 파일로 저장
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    logger.info(f"Dataset prepared: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")

if __name__ == "__main__":
    prepare_dataset() 