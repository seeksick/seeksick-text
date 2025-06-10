import os
import pandas as pd
import glob
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DATASET_PATH = r'C:/Users/Server1/seeksick/KEMDy20_v1_2'

# 텍스트 파일 읽기 및 정제
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='euc-kr') as f:
            text = f.read().strip()
            text = re.sub(r"\b[nbuc]/", "", text)
            text = re.sub(r'[^\w\s가-힣]', '', text)  # 특수문자 제거
            text = text.replace('"', '').replace("'", "")
            return text
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {str(e)}")
        return None

def prepare_kemdy_dataset():
    annotation_files = glob.glob(os.path.join(BASE_DATASET_PATH, 'annotation', '*_eval.csv'))
    logger.info(f"Found {len(annotation_files)} annotation files")
    logger.info(f"Searching in: {os.path.join(BASE_DATASET_PATH, 'annotation')}")
    
    all_data = []
    not_found_files = []
    
    for annotation_file in tqdm(annotation_files, desc="Processing annotation files"):
        try:
            session_data = pd.read_csv(annotation_file, header=[0,1])
            session_name = os.path.basename(annotation_file).split('_')[0]
            session_folder = session_name.replace('Sess', 'Session')
            wav_dir = os.path.join(BASE_DATASET_PATH, 'wav', session_folder)
            
            logger.info(f"Processing session: {session_name}")
            logger.info(f"Found {len(session_data)} segments in {annotation_file}")
            
            for _, row in session_data.iterrows():
                try:
                    segment_id = row[('Segment ID', ' ')]
                except Exception as e:
                    logger.error(f"Failed to get segment ID: {str(e)}")
                    continue
                
                if pd.isna(segment_id):
                    continue
                    
                text_file = os.path.join(wav_dir, f'{segment_id}.txt')
                if not os.path.exists(text_file):
                    if len(not_found_files) < 5:
                        not_found_files.append(text_file)
                    continue
                    
                text = read_text_file(text_file)
                if text is None or text.strip() == '':
                    continue
                    
                # 감정 집계
                emotions = []
                for i in range(1, 11):
                    eval_col = (f'Eval{i:02d}{"F" if i % 2 == 1 else "M"}', 'Emotion')
                    try:
                        emotion = row[eval_col]
                    except KeyError:
                        continue
                    if pd.notna(emotion):
                        # 감정 레이블 정규화
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
                        else:
                            emotions.append(emotion)
                        
                total_emotions = len(emotions)
                if total_emotions == 0:
                    continue
                    
                emotion_probs = {
                    'happy': 0.0,
                    'depressed': 0.0,
                    'surprised': 0.0,
                    'angry': 0.0,
                    'neutral': 0.0
                }
                
                for emotion in emotions:
                    if emotion in emotion_probs:
                        emotion_probs[emotion] += 1.0 / total_emotions
                
                all_data.append({
                    'text': text,
                    'happy': emotion_probs['happy'],
                    'depressed': emotion_probs['depressed'],
                    'surprised': emotion_probs['surprised'],
                    'angry': emotion_probs['angry'],
                    'neutral': emotion_probs['neutral']
                })
                
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
    
    os.makedirs('data', exist_ok=True)
    # 컬럼 순서 지정
    columns = ['text', 'happy', 'depressed', 'surprised', 'angry', 'neutral']
    train_df.to_csv('data/train.csv', index=False, encoding='utf-8', columns=columns)
    val_df.to_csv('data/val.csv', index=False, encoding='utf-8', columns=columns)
    test_df.to_csv('data/test.csv', index=False, encoding='utf-8', columns=columns)
    
    logger.info(f"Dataset prepared: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")

if __name__ == "__main__":
    prepare_kemdy_dataset() 